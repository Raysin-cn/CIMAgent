"""
立场检测器模块

提供社交媒体帖子立场检测功能，支持：
- 单文本立场检测
- 用户立场分析
- 时间演化分析
- 批量处理
"""

import sqlite3
import json
import pandas as pd
import asyncio
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.messages import BaseMessage
from camel.agents import ChatAgent

import sys
import os
# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from cim.config import config
import numpy as np
import matplotlib.pyplot as plt


# 配置日志
logging.basicConfig(level=getattr(logging, config.log_level))
logger = logging.getLogger(__name__)


class StanceDetector:
    """立场检测器"""
    
    def __init__(self, db_path: Optional[str] = None, model_config: Optional[Dict] = None):
        """
        初始化立场检测器
        
        Args:
            db_path: 数据库文件路径，如果为None则使用配置中的默认路径
            model_config: 模型配置，如果为None则使用配置中的默认配置
        """
        self.db_path = db_path or config.database.path
        self.model_config = model_config or config.model.__dict__
        self.model = None
        self.agent = None
        
        # 立场类别定义
        self.stance_categories = ["支持", "反对", "中立", "混合"]
        
        logger.info(f"初始化立场检测器，数据库路径: {self.db_path}")
    
    def _init_model(self):
        """初始化LLM模型"""
        if self.model is None:
            try:
                # 根据配置创建模型
                if self.model_config["platform"].upper() == "VLLM":
                    # 传递 max_tokens 以避免后续警告，并将温度设为 0 以提升一致性
                    self.model = ModelFactory.create(
                        model_platform=ModelPlatformType.VLLM,
                        model_type=self.model_config["model_type"],
                        url=self.model_config["url"],
                        model_config_dict={
                            "temperature": 0.0,
                            "max_tokens": self.model_config.get("max_tokens", 20480)
                        }
                    )
                else:
                    # 默认使用OpenAI
                    self.model = ModelFactory.create(
                        model_platform=ModelPlatformType.OPENAI,
                        model_type=ModelType.GPT_4O_MINI,
                    )
                
                # 创建ChatAgent
                self.agent = ChatAgent(
                    system_message="你是一个社交媒体帖子立场识别助手，能够根据帖子内容和话题识别帖子表达的立场。",
                    model=self.model
                )
                
                logger.info(f"✓ 成功初始化模型: {self.model_config['platform']}")
            except Exception as e:
                logger.error(f"❌ 模型初始化失败: {e}")
                raise
    
    def _get_db_connection(self) -> sqlite3.Connection:
        """获取数据库连接"""
        try:
            return sqlite3.connect(self.db_path)
        except Exception as e:
            logger.error(f"❌ 数据库连接失败: {e}")
            raise
    
    async def get_user_recent_posts(self, user_id: int, limit: int = None) -> List[Dict]:
        """
        获取用户最近发布的帖子
        
        Args:
            user_id: 用户ID
            limit: 获取的帖子数量限制，如果为None则使用配置中的默认值
            
        Returns:
            帖子列表，每个帖子包含post_id, content, created_at等信息
        """
        limit = limit or config.stance.post_limit
        
        # 使用线程池执行数据库操作，避免阻塞主线程
        loop = asyncio.get_running_loop()
        
        def db_operation():
            try:
                conn = self._get_db_connection()
                cursor = conn.cursor()
                
                # 查询用户最近发布的原创帖子（排除转发和引用）
                query = """
                    SELECT post_id, content, created_at, num_likes, num_dislikes, num_shares
                    FROM post 
                    WHERE user_id = ? AND original_post_id IS NULL
                    ORDER BY created_at DESC
                    LIMIT ?
                """
                
                cursor.execute(query, (user_id, limit))
                posts = cursor.fetchall()
                
                # 转换为字典格式
                post_list = []
                for post in posts:
                    post_dict = {
                        'post_id': post[0],
                        'content': post[1],
                        'created_at': post[2],
                        'num_likes': post[3],
                        'num_dislikes': post[4],
                        'num_shares': post[5]
                    }
                    post_list.append(post_dict)
                
                conn.close()
                logger.debug(f"获取用户 {user_id} 的 {len(post_list)} 条帖子")
                return post_list
                
            except Exception as e:
                logger.error(f"❌ 获取用户 {user_id} 的帖子失败: {e}")
                return []
        
        # 在线程池中执行数据库操作
        with ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(executor, db_operation)
    
    async def get_all_users_with_posts(self) -> List[int]:
        """
        获取所有发布过帖子的用户ID列表
        
        Returns:
            用户ID列表
        """
        loop = asyncio.get_running_loop()
        
        def db_operation():
            try:
                conn = self._get_db_connection()
                cursor = conn.cursor()
                
                # 查询所有发布过原创帖子的用户
                query = """
                    SELECT DISTINCT user_id 
                    FROM post 
                    WHERE original_post_id IS NULL AND user_id != -1
                    ORDER BY user_id
                """
                
                cursor.execute(query)
                users = [row[0] for row in cursor.fetchall()]
                
                conn.close()
                logger.info(f"找到 {len(users)} 个发布过帖子的用户")
                return users
                
            except Exception as e:
                logger.error(f"❌ 获取用户列表失败: {e}")
                return []
        
        # 在线程池中执行数据库操作
        with ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(executor, db_operation)
    
    async def get_user_info(self, user_id: int) -> Optional[Dict]:
        """
        获取用户信息
        
        Args:
            user_id: 用户ID
            
        Returns:
            用户信息字典
        """
        loop = asyncio.get_running_loop()
        
        def db_operation():
            try:
                conn = self._get_db_connection()
                cursor = conn.cursor()
                
                query = "SELECT user_id, agent_id, user_name, name, bio FROM user WHERE user_id = ?"
                cursor.execute(query, (user_id,))
                user = cursor.fetchone()
                
                conn.close()
                
                if user:
                    return {
                        'user_id': user[0],
                        'agent_id': user[1],
                        'user_name': user[2],
                        'name': user[3],
                        'bio': user[4]
                    }
                return None
                
            except Exception as e:
                logger.error(f"❌ 获取用户 {user_id} 信息失败: {e}")
                return None
        
        # 在线程池中执行数据库操作
        with ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(executor, db_operation)
    
    async def _generate_stance_prompt(self, text: str, topic: str) -> str:
        """生成立场检测的提示词（带规则与示例，严格JSON输出）"""
        return f"""
你是一个精确的立场分类器，任务是判断下面这段文本针对“{topic}”的立场。

请严格遵循两步判定：
1) 相关性判定（优先级最高）
   - 若文本未明确提及与主题强相关的实体或同义表达（如：新疆棉、Xinjiang cotton、涉疆棉花、对新疆棉的采购/抵制/禁令/制裁等），且仅谈及泛化的伦理/透明/技术/时尚观点，则判定为「无关」。
   - 若提及新疆棉或与其采购/抵制直接相关的内容，进入第2步。

2) 立场判定规则（仅在与主题相关时使用）
   - 「支持」：明确支持采购/继续购买/反对抵制/呼吁在透明与合规前提下继续采购；或将“反对盲目抵制”与“支持在透明审计下采购”联用。
   - 「反对」：明确支持抵制/停止采购/禁令/制裁新疆棉。
   - 「混合」：同时出现支持与反对的要点，或明确表达“既不应抵制也不应盲目支持、应视证据而定”。
   - 「中立」：与主题相关，但仅讨论流程/透明度/技术方案/信息披露/第三方审计等，不对“是否采购/是否抵制”表态。

判定优先级建议：
   无关 > 反对/支持 > 混合 > 中立。

注意：
- 输出必须是单个JSON对象，使用双引号，不要输出任何额外文本、不要使用Markdown代码块、不要输出<think>或其它标签。
- 若难以判定，请在规则内做出最贴近的选择，并通过较低的confidence表达不确定性。

输出格式（严格）：
{{
  "stance": "支持|反对|中立|混合|无关",
  "confidence": 0-1之间的小数,
  "reasoning": "不超过80字的要点式理由",
  "keywords": ["提取到的关键短语..."]
}}

示例（仅作风格参考，不要复述示例文本）：
1) 文本："Reject boycotts; support audited supply chains for Xinjiang cotton."
   输出：{{"stance":"支持","confidence":0.9,"reasoning":"反对抵制并支持在审计下采购","keywords":["reject boycotts","audited"]}}

2) 文本："Stop buying Xinjiang cotton until further notice."
   输出：{{"stance":"反对","confidence":0.95,"reasoning":"明确主张停止采购","keywords":["stop buying"]}}

3) 文本："Not boycotts or blind support—act on evidence."
   输出：{{"stance":"混合","confidence":0.8,"reasoning":"同时反对抵制与盲目支持，取决于证据","keywords":["not boycotts","not blind support"]}}

4) 文本："Use AR/blockchain to show supply chain steps for Xinjiang cotton."
   输出：{{"stance":"中立","confidence":0.7,"reasoning":"仅描述透明/技术方案，无采购/抵制定性","keywords":["AR","blockchain"]}}

5) 文本："I love ethical fashion. Let's know the story behind clothes."
   输出：{{"stance":"无关","confidence":0.8,"reasoning":"泛化伦理观点，未涉及新疆棉或采购/抵制","keywords":["ethical fashion"]}}

现在请基于上述规则对以下文本进行判断：
文本内容：{text}

只返回JSON对象。
        """

    def _extract_json_object(self, content: str) -> Optional[Dict[str, Any]]:
        """从模型输出中稳健提取JSON对象。

        策略：
        - 去除围绕内容的代码围栏/空白
        - 通过括号计数提取第一个完整的JSON对象
        - 解析失败则返回None
        """
        if not content:
            return None

        text = content.strip()
        # 去除常见的Markdown代码围栏
        if text.startswith("```"):
            # 去掉首尾围栏
            lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
            text = "\n".join(lines).strip()

        # 通过括号匹配截取第一个完整的JSON对象
        start_idx = -1
        brace_depth = 0
        for idx, ch in enumerate(text):
            if ch == '{':
                if brace_depth == 0:
                    start_idx = idx
                brace_depth += 1
            elif ch == '}':
                if brace_depth > 0:
                    brace_depth -= 1
                    if brace_depth == 0 and start_idx != -1:
                        candidate = text[start_idx:idx+1]
                        try:
                            return json.loads(candidate)
                        except Exception:
                            # 继续尝试后续片段
                            start_idx = -1
                            brace_depth = 0
        # 直接整体解析的兜底
        try:
            return json.loads(text)
        except Exception:
            return None
    
    async def detect_stance_for_texts_batch(self, texts: List[str], topic: str = None, max_retries: int = None) -> List[Dict]:
        """
        批量检测多个文本的立场
        
        Args:
            texts: 文本列表
            topic: 检测的主题
            max_retries: 最大重试次数
            
        Returns:
            立场检测结果列表
        """
        topic = topic or config.stance.default_topic
        max_retries = max_retries or config.stance.max_retries
        
        if not texts:
            return []
            
        self._init_model()
        
        # 并行处理所有文本
        tasks = [self.detect_stance_for_text(text, topic, max_retries) for text in texts]
        results = await asyncio.gather(*tasks)
        
        return results
    
    async def detect_stance_for_text(self, text: str, topic: str = None, max_retries: int = None) -> Dict:
        """
        对单个文本进行立场检测
        
        Args:
            text: 要检测的文本
            topic: 检测的主题，如果为None则使用配置中的默认主题
            max_retries: 最大重试次数，如果为None则使用配置中的默认值
            
        Returns:
            立场检测结果，包含立场、置信度、理由等
        """
        topic = topic or config.stance.default_topic
        max_retries = max_retries or config.stance.max_retries
        
        self._init_model()
        
        prompt = await self._generate_stance_prompt(text, topic)
        
        for attempt in range(max_retries):
            try:
                response = await self.agent.astep(
                    BaseMessage.make_user_message(
                        role_name="User",
                        content=prompt
                    )
                )
                
                if response.msgs and len(response.msgs) > 0:
                    content = response.msgs[0].content.strip()
                    
                    # 稳健提取JSON（处理模型附带<think>或解释文本的情况）
                    result = self._extract_json_object(content)
                    if result is None:
                        logger.warning(f"无法解析JSON响应: {content}")
                    else:
                        # 验证结果格式
                        if all(key in result for key in ["stance", "confidence", "reasoning"]):
                            result["text"] = text
                            result["topic"] = topic
                            result["detection_time"] = datetime.now().isoformat()
                            logger.debug(f"立场检测成功: {result['stance']} (置信度: {result['confidence']})")
                            return result
                        else:
                            logger.warning(f"立场检测结果格式不完整: {result}")
                        
            except Exception as e:
                logger.warning(f"立场检测尝试 {attempt + 1} 失败: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)  # 重试前等待
        
        # 所有重试都失败，返回默认结果
        logger.error(f"立场检测失败，返回默认结果")
        return {
            "stance": "中立",
            "confidence": 0.0,
            "reasoning": "检测失败，使用默认中立立场",
            "keywords": [],
            "text": text,
            "topic": topic,
            "detection_time": datetime.now().isoformat()
        }
    
    async def detect_stance_for_user(self, user_id: int, topic: str = None, post_limit: int = None) -> Dict:
        """
        检测指定用户的立场
        
        Args:
            user_id: 用户ID
            topic: 检测主题
            post_limit: 帖子数量限制
            
        Returns:
            用户立场检测结果
        """
        topic = topic or config.stance.default_topic
        post_limit = post_limit or config.stance.post_limit
        
        # 并行获取用户信息和最近帖子
        user_info, posts = await asyncio.gather(
            self.get_user_info(user_id),
            self.get_user_recent_posts(user_id, post_limit)
        )
        
        if not user_info:
            logger.warning(f"用户 {user_id} 不存在")
            return {"error": f"用户 {user_id} 不存在"}
        
        if not posts:
            logger.warning(f"用户 {user_id} 没有发布过帖子")
            return {"error": f"用户 {user_id} 没有发布过帖子"}
        
        # 提取帖子内容用于批量检测，并保留时间戳
        post_contents = [post['content'] for post in posts]
        post_ids = [post['post_id'] for post in posts]
        post_created_ats = [post['created_at'] for post in posts]
        
        # 批量检测所有帖子的立场
        stance_results = await self.detect_stance_for_texts_batch(post_contents, topic)
        
        # 添加帖子ID和内容
        post_stances = []
        for i, stance in enumerate(stance_results):
            stance['post_id'] = post_ids[i]
            stance['post_content'] = post_contents[i]
            # 将原帖发布时间写入结果，供后续时序分析
            stance['created_at'] = post_created_ats[i]
            post_stances.append(stance)
        
        # 综合用户立场
        stance_counts = {}
        total_confidence = 0
        
        for stance in post_stances:
            stance_type = stance['stance']
            stance_counts[stance_type] = stance_counts.get(stance_type, 0) + 1
            total_confidence += float(stance['confidence'])
        
        # 确定主要立场
        main_stance = max(stance_counts.items(), key=lambda x: x[1])[0]
        avg_confidence = total_confidence / len(post_stances) if post_stances else 0
        
        result = {
            "user_id": user_id,
            "user_name": user_info.get('user_name', ''),
            "name": user_info.get('name', ''),
            "topic": topic,
            "posts_analyzed": len(posts),
            "main_stance": main_stance,
            "stance_distribution": stance_counts,
            "average_confidence": avg_confidence,
            "post_stances": post_stances,
            "analysis_time": datetime.now().isoformat()
        }
        
        logger.info(f"用户 {user_id} 立场检测完成: {main_stance}")
        return result
    
    async def detect_stance_for_all_users(self, topic: str = None, post_limit: int = None) -> List[Dict]:
        """
        检测所有用户的立场
        
        Args:
            topic: 检测主题
            post_limit: 每个用户的帖子数量限制
            
        Returns:
            所有用户的立场检测结果列表
        """
        topic = topic or config.stance.default_topic
        post_limit = post_limit or config.stance.post_limit
        
        users = await self.get_all_users_with_posts()
        if not users:
            logger.warning("没有找到发布过帖子的用户")
            return []
        
        logger.info(f"开始检测 {len(users)} 个用户的立场")
        
        # 并发检测所有用户
        semaphore = asyncio.Semaphore(config.stance.max_concurrent)
        
        async def process_user(user_id):
            async with semaphore:
                return await self.detect_stance_for_user(user_id, topic, post_limit)
        
        tasks = [process_user(user_id) for user_id in users]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 过滤掉错误结果
        valid_results = [r for r in results if isinstance(r, dict) and "error" not in r]
        
        logger.info(f"立场检测完成，成功检测 {len(valid_results)} 个用户")
        return valid_results
    
    async def save_stance_results(self, results: List[Dict], output_path: str):
        """
        保存立场检测结果
        
        Args:
            results: 检测结果列表
            output_path: 输出文件路径
        """
        loop = asyncio.get_running_loop()
        
        try:
            # 在线程池中执行IO操作
            with ThreadPoolExecutor() as executor:
                # 保存为JSON
                await loop.run_in_executor(
                    executor,
                    lambda: self._save_json_results(results, output_path)
                )
                
                # 保存为CSV
                csv_path = output_path.replace('.json', '.csv')
                await loop.run_in_executor(
                    executor,
                    lambda: self._save_csv_results(results, csv_path)
                )
                
            logger.info(f"结果已保存: {output_path}, {csv_path}")
            
        except Exception as e:
            logger.error(f"❌ 保存结果失败: {e}")
    
    def _save_json_results(self, results: List[Dict], output_path: str):
        """保存为JSON文件"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    
    def _save_csv_results(self, results: List[Dict], csv_path: str):
        """保存为CSV文件"""
        csv_data = []
        
        for result in results:
            if "error" in result:
                continue
                
            csv_data.append({
                'user_id': result['user_id'],
                'user_name': result['user_name'],
                'name': result['name'],
                'topic': result['topic'],
                'posts_analyzed': result['posts_analyzed'],
                'main_stance': result['main_stance'],
                'average_confidence': result['average_confidence'],
                'analysis_time': result['analysis_time']
            })
        
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_path, index=False, encoding='utf-8')
    
    def generate_stance_summary(self, results: List[Dict]) -> Dict:
        """
        生成立场检测结果摘要
        
        Args:
            results: 检测结果列表
            
        Returns:
            摘要信息
        """
        if not results:
            return {"error": "没有检测结果"}
        
        valid_results = [r for r in results if "error" not in r]
        
        if not valid_results:
            return {"error": "没有有效的检测结果"}
        
        # 统计信息
        total_users = len(valid_results)
        total_posts = sum(r['posts_analyzed'] for r in valid_results)
        avg_confidence = sum(r['average_confidence'] for r in valid_results) / total_users
        
        # 立场分布
        stance_distribution = {}
        for result in valid_results:
            stance = result['main_stance']
            stance_distribution[stance] = stance_distribution.get(stance, 0) + 1
        
        # 最常见立场
        most_common_stance = max(stance_distribution.items(), key=lambda x: x[1])[0]
        
        summary = {
            "total_users": total_users,
            "valid_users": len(valid_results),
            "total_posts_analyzed": total_posts,
            "average_confidence": avg_confidence,
            "most_common_stance": most_common_stance,
            "stance_distribution": stance_distribution,
            "summary_time": datetime.now().isoformat()
        }
        
        return summary 

    def plot_users_stance_evolution(
        self,
        results: List[Dict],
        save_path: Optional[str] = None,
        alpha: float = 0.6,
        smoothing_alpha: float = 0.6,
        line_separation: float = 0.06,
    ) -> Optional[str]:
        """绘制用户立场动态演化图。

        约定与设计：
        - 时间步：对每个用户，按照帖子 `created_at` 升序，将其自身发帖序号作为时间步（1,2,3,...）。
        - 立场数值映射：反对=-1，中立=0，支持=1，混合=0。
        - 立场分值：指数平滑分数 score_t = smoothing_alpha*score_{t-1} + (1-smoothing_alpha)*base_t。
        - 图像：每位用户一条灰色折线，并在各时间步处打点标记该用户发帖。为减少重叠，对每个用户曲线在纵轴施加微小偏移（line_separation）。

        Args:
            results: detect_stance_for_all_users 的输出列表
            save_path: 图片保存路径（.png/.pdf等）；若为None则不保存
            alpha: 折线透明度
            smoothing_alpha: 历史平滑系数，越大历史权重越高

        Returns:
            保存路径（若保存），否则None
        """
        if not results:
            return None

        # 过滤有效用户
        valid_results = [r for r in results if isinstance(r, dict) and "error" not in r and r.get("post_stances")]
        if not valid_results:
            return None

        def stance_to_base_value(stance: str) -> float:
            if stance == "支持":
                return 1.0
            if stance == "反对":
                return -1.0
            # 「中立」「混合」均视为0（无明显方向）
            return 0.0

        plt.figure(figsize=(12, 7))
        ax = plt.gca()

        for idx, user_result in enumerate(valid_results):
            user_posts = user_result.get("post_stances", [])
            # 排序：按 created_at 升序
            try:
                df_user = pd.DataFrame(user_posts)
                # 兼容 created_at 可能为None/空
                df_user["created_at_dt"] = pd.to_datetime(df_user.get("created_at", pd.NaT), errors="coerce")
                df_user = df_user.sort_values(by=["created_at_dt", "post_id"], ascending=[True, True])
            except Exception:
                # 回退：按索引顺序
                df_user = pd.DataFrame(user_posts)

            # 构建时间步与分值
            base_values = [stance_to_base_value(str(s)) for s in df_user.get("stance", []).tolist()]
            if not base_values:
                continue

            scores = []
            prev = 0.0
            for b in base_values:
                prev = smoothing_alpha * prev + (1.0 - smoothing_alpha) * b
                scores.append(prev)

            timesteps = list(range(1, len(scores) + 1))
            # 对每位用户施加细小的纵向偏移，减少曲线重叠
            # 偏移范围与用户索引相关，中心对称分布
            if line_separation and line_separation > 0:
                center = (len(valid_results) - 1) / 2.0
                offset = (idx - center) * line_separation
                y_vals = np.clip(np.array(scores) + offset, -1.05, 1.05)
            else:
                y_vals = np.array(scores)

            ax.plot(timesteps, y_vals, color="gray", alpha=alpha, linewidth=1.2)
            ax.scatter(timesteps, y_vals, color="gray", s=10, alpha=min(alpha + 0.2, 1.0))

        # 坐标轴与标注
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Stance score")
        ax.set_title("Users' stance evolution (gray lines; dots mark posts)")
        ax.set_ylim(-1.05, 1.05)
        ax.set_yticks([-1.0, 0.0, 1.0])
        ax.set_yticklabels(["Oppose (-1)", "Neutral/Mixed (0)", "Support (1)"])
        ax.grid(True, linestyle=":", alpha=0.4)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.close()
        return save_path
    


if __name__ == "__main__":
    import argparse
    import asyncio
    
    async def main():
        """立场检测器主函数"""
        parser = argparse.ArgumentParser(description="用户立场检测分析")
        parser.add_argument("--db_path", type=str, default="./data/processed/twitter_simulation.db",
                           help="数据库文件路径")
        parser.add_argument("--output", type=str, default="./data/processed/stance_detection_results.json",
                           help="输出结果文件路径")
        parser.add_argument("--topic", type=str, default=None,
                           help="检测主题，默认使用配置中的主题")
        parser.add_argument("--post_limit", type=int, default=10,
                           help="每个用户检测的帖子数量限制，默认使用配置中的值")
        parser.add_argument("--evolution", action="store_true",
                           help="启用立场演化分析模式")
        parser.add_argument("--user_id", type=int, default=None,
                           help="指定用户ID进行检测，不指定则检测所有用户")
        
        args = parser.parse_args()
        
        print("=" * 50)
        print("用户立场检测分析")
        print("=" * 50)
        
        # 初始化检测器
        detector = StanceDetector(args.db_path)
        
        try:
            if args.evolution:
                # 立场演化分析模式
                print("🔍 开始立场演化分析...")
                evolution_results = await detector.analyze_stance_evolution(
                    args.topic, args.post_limit
                )
                
                if isinstance(evolution_results, dict) and "error" in evolution_results:
                    print(f"❌ 演化分析失败: {evolution_results['error']}")
                    return
                
                # 保存演化结果
                output_path = args.output.replace('.json', '_evolution.json')
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(evolution_results, f, ensure_ascii=False, indent=2)
                
                print(f"✓ 演化分析结果已保存到: {output_path}")
                
                # 生成CSV格式的演化数据
                csv_data = []
                for user_id, user_evolution in evolution_results['user_evolution'].items():
                    for stance_info in user_evolution['timestep_stances']:
                        csv_data.append({
                            'user_id': user_id,
                            'user_name': user_evolution['user_name'],
                            'name': user_evolution['name'],
                            'timestep': stance_info['timestep'],
                            'posts_analyzed': stance_info['posts_analyzed'],
                            'stance': stance_info.get('stance', stance_info.get('overall_stance', '未知')),
                            'confidence': stance_info.get('confidence', 0.0),
                            'reasoning': stance_info.get('reasoning', '')
                        })
                
                csv_path = output_path.replace('.json', '.csv')
                df = pd.DataFrame(csv_data)
                df.to_csv(csv_path, index=False, encoding='utf-8')
                print(f"✓ 演化数据CSV已保存到: {csv_path}")
                
                # 输出摘要
                print("\n📈 立场演化分析摘要:")
                print(f"- 分析主题: {evolution_results['topic']}")
                print(f"- 时间步数量: {len(evolution_results['timesteps'])}")
                print(f"- 用户数量: {evolution_results['total_users']}")
                
            else:
                # 标准模式: 分析当前立场
                if args.user_id:
                    # 分析指定用户
                    print(f"🔍 检测用户 {args.user_id} 的立场...")
                    result = await detector.detect_stance_for_user(args.user_id, args.topic, args.post_limit)
                    results = [result]
                else:
                    # 分析所有用户
                    print("🔍 检测所有用户的立场...")
                    results = await detector.detect_stance_for_all_users(args.topic, args.post_limit)
                
                # 保存结果
                await detector.save_stance_results(results, args.output)
                
                # 绘制立场演化图（基于每个用户的发帖时间序列）
                try:
                    evolution_fig_path = args.output.replace('.json', '_evolution.png')
                    detector.plot_users_stance_evolution(results, save_path=evolution_fig_path, alpha=0.7)
                    print(f"✓ 立场演化图已保存到: {evolution_fig_path}")
                except Exception as e:
                    print(f"⚠️ 立场演化绘图失败: {e}")
                
                # 生成摘要
                summary = detector.generate_stance_summary(results)
                print("\n📊 检测结果摘要:")
                print(f"- 总用户数: {summary.get('total_users', 0)}")
                print(f"- 有效用户数: {summary.get('valid_users', 0)}")
                print(f"- 分析帖子总数: {summary.get('total_posts_analyzed', 0)}")
                print(f"- 平均置信度: {summary.get('average_confidence', 0.0):.2f}")
                print(f"- 最常见立场: {summary.get('most_common_stance', '未知')}")
                print("\n立场分布:")
                for stance, count in summary.get('stance_distribution', {}).items():
                    print(f"  {stance}: {count} 人")
                
        except Exception as e:
            print(f"❌ 检测过程出错: {e}")
            import traceback
            traceback.print_exc()
    
    # 运行主函数
    asyncio.run(main())
    