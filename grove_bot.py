"""Python 3.11+
Install: python -m pip install -U discord.py openai python-dotenv
Create .env alongside this script (never commit or share it):
DISCORD_TOKEN=your_discord_bot_token
PGS_API_KEY=your_phoenix_grove_key
PGS_MODEL=glm-5.2
# Optional: sync commands immediately to a development server:
# DISCORD_GUILD_ID=your_server_id

Run: python grove_bot.py
Commands: /ask prompt, /reset
Conversations live in memory, separately per user and channel.
"""

import asyncio
import io
import logging
import os
from collections import OrderedDict
from pathlib import Path

import discord
from discord import app_commands
from dotenv import load_dotenv
from openai import AsyncOpenAI, APIConnectionError, APIStatusError, APITimeoutError

load_dotenv(Path(__file__).with_name('.env'))
logging.basicConfig(level=logging.INFO)
log = logging.getLogger('grove_bot')
TOKEN = os.getenv('DISCORD_TOKEN', '').strip()
API_KEYS = list(dict.fromkeys(
    key.strip() for key in
    (os.getenv('PGS_API_KEYS') or os.getenv('PGS_API_KEY', '')).split(',')
    if key.strip()
))
MODEL = os.getenv('PGS_MODEL', 'glm-5.2')
GUILD_ID = os.getenv('DISCORD_GUILD_ID', '').strip()
SYSTEM = 'You are a helpful Discord assistant. Be clear and concise.'
MAX_SESSIONS = 200
MAX_MESSAGES = 12


ALLOWED_CHANNEL_ID = 1512306846713643159


class ChannelRestrictedTree(app_commands.CommandTree):
    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.channel_id != ALLOWED_CHANNEL_ID:
            await interaction.response.send_message(
                f'Use this bot only in <#{ALLOWED_CHANNEL_ID}>.',
                ephemeral=True,
                allowed_mentions=discord.AllowedMentions.none(),
            )
            return False
        return True


class GroveBot(discord.Client):
    def __init__(self):
        super().__init__(intents=discord.Intents.default(),
                         allowed_mentions=discord.AllowedMentions.none())
        self.tree = ChannelRestrictedTree(self)
        self.apis = [AsyncOpenAI(
            api_key=key, base_url='https://api.pgsgrove.com/v1',
            timeout=90.0, max_retries=0) for key in API_KEYS]
        self.next_key = 0
        self.history = OrderedDict()
        self.busy = set()
        self.slots = asyncio.Semaphore(3)

    async def setup_hook(self):
        if GUILD_ID:
            guild = discord.Object(id=int(GUILD_ID))
            self.tree.copy_global_to(guild=guild)
            await self.tree.sync(guild=guild)
        else:
            await self.tree.sync()

    async def on_ready(self):
        log.info('Connected as %s. Model: %s', self.user, MODEL)

    async def start(self, token, *, reconnect=True):
        from aiohttp import web

        async def health(request):
            return web.json_response({
                'status': 'running',
                'discord_connected': self.is_ready(),
            })

        app = web.Application()
        app.router.add_get('/', health)
        app.router.add_get('/health', health)
        runner = web.AppRunner(app)
        await runner.setup()
        try:
            port = int(os.getenv('PORT', '10000'))
            await web.TCPSite(runner, '0.0.0.0', port).start()
            log.info('HTTP health endpoint listening on port %s', port)
            await super().start(token, reconnect=reconnect)
        finally:
            await runner.cleanup()

    async def close(self):
        await asyncio.gather(*(api.close() for api in self.apis))
        await super().close()


# Keep initialization errors readable; never log credentials.
if not TOKEN or not API_KEYS:
    raise SystemExit('Set DISCORD_TOKEN and PGS_API_KEYS in Render or your .env file.')
if GUILD_ID and not GUILD_ID.isdigit():
    raise SystemExit('DISCORD_GUILD_ID must be a numeric server ID.')

bot = GroveBot()


@bot.tree.command(name='ask', description='Ask AI through Phoenix Grove')
@app_commands.describe(prompt='Your question (sent to Phoenix Grove)')
@app_commands.checks.cooldown(1, 10.0, key=lambda i: i.user.id)
async def ask(interaction: discord.Interaction, prompt: str):
    if not prompt.strip() or len(prompt) > 6000:
        await interaction.response.send_message(
            'Enter a question between 1 and 6,000 characters.', ephemeral=True)
        return
    # A user can have only one in-flight request across all channels.
    if interaction.user.id in bot.busy:
        await interaction.response.send_message(
            'Wait for your current answer to finish.', ephemeral=True)
        return
    key = (interaction.guild_id, interaction.channel_id, interaction.user.id)
    bot.busy.add(interaction.user.id)
    try:
        # Private replies avoid publishing user questions and AI responses.
        await interaction.response.defer(thinking=True, ephemeral=True)
        conversation = list(bot.history.get(key, []))
        conversation.append({'role': 'user', 'content': prompt})
        async with asyncio.timeout(120):
            async with bot.slots:
                api = bot.apis[bot.next_key]
                bot.next_key = (bot.next_key + 1) % len(bot.apis)
                response = await api.chat.completions.create(
                    model=MODEL,
                    messages=[{'role': 'system', 'content': SYSTEM}] + conversation,
                    max_tokens=1500,
                )
        answer = (response.choices[0].message.content or '').strip()
        if not answer:
            await interaction.followup.send(
                'The model returned no text. Try rephrasing your question.',
                ephemeral=True)
            return
        conversation.append({'role': 'assistant', 'content': answer})
        bot.history[key] = conversation[-MAX_MESSAGES:]
        bot.history.move_to_end(key)
        while len(bot.history) > MAX_SESSIONS:
            bot.history.popitem(last=False)
        if len(answer) <= 1900:
            await interaction.followup.send(answer, ephemeral=True,
                allowed_mentions=discord.AllowedMentions.none())
        else:
            await interaction.followup.send(
                'Your answer is attached because it exceeds Discord’s message limit.',
                file=discord.File(io.BytesIO(answer.encode('utf-8')),
                                  filename='answer.txt'), ephemeral=True)
    except (APITimeoutError, asyncio.TimeoutError):
        await interaction.followup.send('The request timed out. Try again shortly.',
                                        ephemeral=True)
    except APIConnectionError:
        await interaction.followup.send('Could not connect to Phoenix Grove.',
                                        ephemeral=True)
    except APIStatusError as exc:
        messages = {
            401: 'Phoenix Grove rejected the API key. Ask the bot owner to check it.',
            402: 'The Phoenix Grove account needs additional credits.',
            403: 'Phoenix Grove denied access to this request or model.',
            429: 'Phoenix Grove is rate limiting requests. Try again shortly.',
        }
        await interaction.followup.send(messages.get(exc.status_code,
            f'Phoenix Grove returned HTTP {exc.status_code}. Check the model and account.'),
            ephemeral=True)
    except Exception as exc:
        log.error('Request failed (%s)', type(exc).__name__)
        if interaction.response.is_done():
            await interaction.followup.send('Unable to complete this request.',
                                            ephemeral=True)
        else:
            await interaction.response.send_message('Unable to complete this request.',
                                                     ephemeral=True)
    finally:
        bot.busy.discard(interaction.user.id)


@bot.tree.command(name='reset', description='Clear your AI conversation in this channel')
async def reset(interaction: discord.Interaction):
    if interaction.user.id in bot.busy:
        await interaction.response.send_message('Wait for your answer, then reset.',
                                                 ephemeral=True)
        return
    key = (interaction.guild_id, interaction.channel_id, interaction.user.id)
    bot.history.pop(key, None)
    await interaction.response.send_message('Your conversation has been cleared.',
                                             ephemeral=True)


@bot.tree.error
async def command_error(interaction: discord.Interaction,
                        error: app_commands.AppCommandError):
    text = (f'Please wait {error.retry_after:.0f} seconds before asking again.'
            if isinstance(error, app_commands.CommandOnCooldown)
            else 'The command failed. Try again shortly.')
    if interaction.response.is_done():
        await interaction.followup.send(text, ephemeral=True)
    else:
        await interaction.response.send_message(text, ephemeral=True)


if __name__ == '__main__':
    bot.run(TOKEN)
