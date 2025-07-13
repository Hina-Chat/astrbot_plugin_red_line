import asyncio
import json

import os
import random
import time
from collections import defaultdict
from typing import Dict

import aiofiles

from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.event.filter import EventMessageType
from astrbot.api.message_components import Plain
from astrbot.api.provider import LLMResponse
from astrbot.api import logger
from astrbot.api.star import Context, Star, StarTools


class RedLine(Star):
    def __init__(self, context: Context, config: dict):
        super().__init__(context)

        general_config = config.get("general", {})

        self.keywords = [kw.lower() for kw in general_config.get("keywords", [])]
        self.threshold = general_config.get("threshold", 3)
        self.probability = general_config.get("probability", 0.5)
        self.release_seconds = general_config.get("release_hours", 6.0) * 3600
        # 取得發佈時重設標誌
        self.reset_on_release = general_config.get("reset_on_release", True)
        self.whitelist = [str(uid) for uid in general_config.get("whitelist", [])]

        data_file_name = general_config.get("data_file", "red_line_data.json")

        # 從目錄名稱可靠地取得插件名稱
        plugin_root_path = os.path.dirname(os.path.abspath(__file__))
        plugin_name = os.path.basename(plugin_root_path)

        # 使用 StarTools 取得插件的標準化資料目錄
        data_dir = StarTools.get_data_dir(plugin_name)
        data_dir.mkdir(parents=True, exist_ok=True)
        self.data_file_path = data_dir / data_file_name

        self.user_data: Dict[str, Dict] = {}
        self.user_locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._load_user_data()

        logger.info("RedLine plugin loaded.")
        logger.info(f"Keywords: {self.keywords}")
        logger.info(f"Threshold: {self.threshold}, Probability: {self.probability}")
        logger.info(f"Release Time: {self.release_seconds / 3600} hours")
        logger.info(f"Reset on Release: {self.reset_on_release}")

    def _load_user_data(self):
        """Loads user violation data. Handles migration from old format."""
        try:
            if os.path.exists(self.data_file_path):
                with open(self.data_file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for user_id, value in data.items():
                    if isinstance(value, int):
                        self.user_data[user_id] = {
                            "count": value,
                            "last_triggered": time.time(),
                        }
                    elif isinstance(value, dict):
                        self.user_data[user_id] = value
                logger.info(f"Loaded and migrated user data from {self.data_file_path}")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(
                f"Could not load user data file, starting fresh. Reason: {e}"
            )
            self.user_data = {}

    async def _save_user_data(self):
        """Saves user violation data to the data file asynchronously."""
        try:
            async with aiofiles.open(self.data_file_path, "w", encoding="utf-8") as f:
                await f.write(json.dumps(self.user_data, indent=4))
            logger.debug(f"Saved user data to {self.data_file_path}")
        except IOError as e:
            logger.error(f"Failed to save user data. Reason: {e}")

    async def terminate(self):
        logger.info("RedLine plugin is terminating, saving data...")
        await self._save_user_data()

    @filter.on_llm_response()
    async def on_llm_response(self, event: AstrMessageEvent, response: LLMResponse):
        """根據 LLM 的最終回應來計算違規行為。"""
        user_id = str(event.get_sender_id())
        if user_id in self.whitelist:
            return

        if not self.keywords:
            return

        text_response = response.completion_text
        if not text_response:
            return
        text_response = text_response.lower()

        hit_count = sum(text_response.count(kw) for kw in self.keywords)
        if hit_count > 0:
            async with self.user_locks[user_id]:
                user_record = self.user_data.get(
                    user_id, {"count": 0, "last_triggered": 0}
                )
                user_record["count"] += hit_count
                user_record["last_triggered"] = int(time.time())
                # Store the session_id that caused the violation
                user_record["offending_session_id"] = event.get_session_id()
                self.user_data[user_id] = user_record
                await self._save_user_data()
                logger.info(
                    f"User {user_id} triggered keywords {hit_count} times. New total: {user_record['count']}"
                )

    @filter.event_message_type(EventMessageType.ALL, priority=5)
    async def on_user_message(self, event: AstrMessageEvent):
        """在處理使用者訊息之前檢查其狀態。
        它負責釋放用戶（冷卻）並阻止用戶，支援非同步 I/O 和並發控制。
        """
        user_id = str(event.get_sender_id())
        if user_id in self.whitelist:
            return

        # 在這裡獲取 Lock，以確保對此事件處理程序對該使用者資料的所有後續操作都是原子的。
        async with self.user_locks[user_id]:
            user_record = self.user_data.get(user_id)

            if not user_record or user_record.get("count", 0) == 0:
                return  # 使用者是乾淨的，無需執行任何操作。

            current_time = time.time()
            last_triggered_time = user_record.get("last_triggered", 0)

            # 1. 檢查釋放（冷卻）
            if current_time - last_triggered_time > self.release_seconds:
                logger.info(
                    f"User {user_id} cooldown expired. Resetting count from {user_record['count']} to 0."
                )
                user_record["count"] = 0
                user_record["last_triggered"] = 0

                # 檢查釋放（冷卻）
                if self.reset_on_release:
                    # 使用者違規期間儲存的違規 session_id
                    offending_session_id = user_record.get("offending_session_id")
                    if offending_session_id:
                        logger.info(
                            f"Resetting offending conversation {offending_session_id} for released user {user_id}."
                        )
                        try:
                            cid = await self.context.conversation_manager.get_curr_conversation_id(
                                offending_session_id
                            )
                            if cid:
                                await self.context.conversation_manager.update_conversation(
                                    offending_session_id, cid, []
                                )
                                logger.info(
                                    f"Successfully reset conversation {cid} for session {offending_session_id}."
                                )
                                # reset 確認傳送至使用者目前處於活動狀態的 session
                                await event.send(MessageChain([Plain("……")]))
                            else:
                                logger.warning(
                                    f"Attempted to reset conversation for session {offending_session_id}, but no active conversation found."
                                )
                        except Exception as e:
                            logger.error(
                                f"Failed to reset conversation for released user {user_id} in session {offending_session_id}: {e}",
                                exc_info=True,
                            )
                    else:
                        logger.warning(
                            f"User {user_id} is being released, but no 'offending_session_id' was found in their record. Cannot reset conversation."
                        )

                self.user_data[user_id] = user_record
                await self._save_user_data()
                return  # 使用者已放行，放行訊息。

            # 2. Check for Blocking (如果未釋放)
            # 此檢查現在位於 lock 內，確保讀取最新的計數。
            if user_record.get("count", 0) >= self.threshold:
                if random.random() < self.probability:
                    logger.warning(
                        f"Blocking message from user {user_id} due to violation count ({user_record['count']}/{self.threshold})."
                    )
                    event.stop_event()
                    # BOT 維持靜默規則，如果不刪除注釋。
                    # await event.send(MessageChain([Plain("您的消息因觸發審核規則而被攔截。")]))
