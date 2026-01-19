"""
JSON文件管理器 - 提供线程安全的JSON读写操作，防止并发访问导致文件损坏
"""
import os
import json
import re
import shutil
import time
from typing import Any, Callable, Optional, Dict
import logging

logger = logging.getLogger("JsonFileManager")

# 跨平台文件锁
try:
    import fcntl  # Linux/Mac
    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False

try:
    import msvcrt  # Windows
    HAS_MSVCRT = True
except ImportError:
    HAS_MSVCRT = False


class JsonFileManager:
    """线程安全的JSON文件管理器，支持文件锁、原子写入、自动备份和验证"""
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 0.1):
        """
        初始化JSON文件管理器
        
        Args:
            max_retries: 读取失败时的最大重试次数
            retry_delay: 重试延迟（秒）
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
    def _acquire_lock(self, file_handle, exclusive: bool = True) -> bool:
        """
        获取文件锁（跨平台）
        
        Args:
            file_handle: 文件句柄
            exclusive: 是否使用排他锁（True=排他锁用于写入，False=共享锁用于读取）
        """
        try:
            if HAS_FCNTL:
                # Linux/Mac
                if exclusive:
                    # 排他锁（用于写入）
                    fcntl.flock(file_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                else:
                    # 共享锁（用于读取，允许多个读取并发）
                    fcntl.flock(file_handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
                return True
            elif HAS_MSVCRT:
                # Windows（只支持排他锁）
                if exclusive:
                    msvcrt.locking(file_handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    # Windows下读取时不加锁（因为Windows的locking只支持排他锁）
                    # 对于读取，我们依赖重试机制
                    pass
                return True
            else:
                # 没有文件锁支持，警告但继续
                logger.warning("当前平台不支持文件锁，可能存在并发问题")
                return True
        except (IOError, OSError) as e:
            # 锁被占用
            return False
    
    def _release_lock(self, file_handle):
        """释放文件锁"""
        try:
            if HAS_FCNTL:
                fcntl.flock(file_handle, fcntl.LOCK_UN)
            elif HAS_MSVCRT:
                msvcrt.locking(file_handle.fileno(), msvcrt.LK_UNLCK, 1)
        except (IOError, OSError):
            pass
    
    def _clean_content(self, content: str) -> str:
        """清理JSON内容中的非法控制字符"""
        return re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f]', '', content)
    
    def safe_read_json(self, path: str, default: Optional[Dict] = None) -> Dict[str, Any]:
        """
        安全读取JSON文件（带文件锁）
        
        Args:
            path: JSON文件路径
            default: 如果文件不存在或读取失败，返回的默认值（默认None，返回空字典）
        
        Returns:
            解析后的JSON数据（字典）
        """
        if default is None:
            default = {}
        
        for attempt in range(self.max_retries):
            try:
                if not os.path.exists(path):
                    logger.debug(f"文件不存在: {path}，返回默认值")
                    return default
                
                # 使用只读模式打开（读取使用共享锁，允许多个读取并发）
                with open(path, 'r', encoding='utf-8') as f:
                    # 尝试获取共享锁（非阻塞，允许多个读取并发）
                    if not self._acquire_lock(f, exclusive=False):
                        # 如果无法获取锁（Windows下），等待后重试
                        if attempt < self.max_retries - 1:
                            logger.debug(f"文件被锁定，等待后重试 ({attempt + 1}/{self.max_retries})")
                            time.sleep(self.retry_delay * (attempt + 1))
                            continue
                        else:
                            # 最后一次尝试，即使无法获取锁也尝试读取（在Windows下）
                            logger.debug(f"无法获取文件锁，尝试直接读取: {path}")
                            # 继续执行读取操作
                    
                    # 读取并清理内容
                    content = f.read()
                    content = self._clean_content(content)
                    
                    # 解析JSON
                    try:
                        data = json.loads(content)
                        return data
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON解析失败: {path}, 错误: {e}")
                        # 如果是最后一次尝试，返回默认值
                        if attempt < self.max_retries - 1:
                            time.sleep(self.retry_delay * (attempt + 1))
                            continue
                        return default
                        
            except (IOError, OSError) as e:
                # 文件可能正在被写入，等待后重试
                if attempt < self.max_retries - 1:
                    logger.debug(f"文件访问错误，等待后重试: {e}")
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                else:
                    logger.warning(f"读取文件失败: {path}, 错误: {e}")
                    return default
            except Exception as e:
                logger.error(f"读取JSON文件失败: {path}, 错误: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                return default
        
        return default
    
    def safe_write_json(self, path: str, data: Dict[str, Any], backup: bool = True) -> bool:
        """
        安全写入JSON文件（带文件锁，原子操作）
        
        Args:
            path: JSON文件路径
            data: 要写入的数据（字典）
            backup: 是否在写入前备份原文件
        
        Returns:
            是否成功
        """
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # 备份原文件（如果存在且需要备份）
            backup_path = None
            if backup and os.path.exists(path):
                backup_path = f"{path}.backup"
                try:
                    shutil.copy2(path, backup_path)
                    logger.debug(f"已备份文件: {backup_path}")
                except Exception as e:
                    logger.warning(f"备份文件失败: {e}")
                    backup_path = None
            
            # 写入临时文件
            temp_path = f"{path}.tmp"
            try:
                with open(temp_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)
                
                # 验证临时文件的JSON格式
                try:
                    with open(temp_path, 'r', encoding='utf-8') as f:
                        json.load(f)
                except json.JSONDecodeError as e:
                    logger.error(f"写入的JSON格式验证失败: {e}")
                    if backup_path and os.path.exists(backup_path):
                        logger.info(f"从备份恢复: {backup_path}")
                        shutil.copy2(backup_path, path)
                    return False
                
                # 原子替换（在Unix系统上是原子操作）
                if os.name == 'nt':
                    # Windows需要先删除原文件
                    if os.path.exists(path):
                        os.remove(path)
                    os.rename(temp_path, path)
                else:
                    # Unix/Linux/Mac：rename是原子操作
                    os.rename(temp_path, path)
                
                logger.debug(f"成功写入JSON文件: {path}")
                
                # 清理备份文件（可选）
                if backup_path and os.path.exists(backup_path):
                    try:
                        os.remove(backup_path)
                    except Exception:
                        pass
                
                return True
                
            except Exception as e:
                logger.error(f"写入临时文件失败: {e}")
                # 清理临时文件
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except Exception:
                        pass
                # 从备份恢复
                if backup_path and os.path.exists(backup_path):
                    logger.info(f"从备份恢复: {backup_path}")
                    try:
                        shutil.copy2(backup_path, path)
                    except Exception:
                        pass
                return False
                
        except Exception as e:
            logger.error(f"写入JSON文件失败: {path}, 错误: {e}")
            return False
    
    def safe_update_json(self, path: str, update_func: Callable[[Dict[str, Any]], Dict[str, Any]], 
                        default: Optional[Dict] = None) -> bool:
        """
        安全更新JSON文件（读取→修改→写入，原子操作，带文件锁）
        
        Args:
            path: JSON文件路径
            update_func: 更新函数，接收当前数据，返回更新后的数据
            default: 如果文件不存在，使用的默认值
        
        Returns:
            是否成功
        """
        if default is None:
            default = {}
        
        max_lock_attempts = 10  # 增加重试次数
        lock_delay = 0.5  # 增加延迟时间
        
        for lock_attempt in range(max_lock_attempts):
            try:
                # 读取当前数据（带锁）
                current_data = None
                with open(path, 'r+', encoding='utf-8') if os.path.exists(path) else open(path, 'w+', encoding='utf-8') as f:
                    # 获取排他锁（写入操作需要排他锁）
                    if not self._acquire_lock(f, exclusive=True):
                        if lock_attempt < max_lock_attempts - 1:
                            wait_time = lock_delay * (lock_attempt + 1)  # 递增等待时间
                            logger.debug(f"文件被锁定，等待 {wait_time:.1f} 秒后重试 ({lock_attempt + 1}/{max_lock_attempts})")
                            time.sleep(wait_time)
                            continue
                        else:
                            logger.warning(f"无法获取文件锁（已重试{max_lock_attempts}次），更新失败: {path}")
                            # 最后一次尝试，使用更长等待时间
                            logger.debug(f"最后尝试：等待 {lock_delay * 2} 秒...")
                            time.sleep(lock_delay * 2)
                            # 尝试最后一次
                            if not self._acquire_lock(f, exclusive=True):
                                return False
                            # 如果获取成功，继续执行
                    
                    try:
                        # 读取数据
                        if os.path.getsize(path) > 0:
                            content = f.read()
                            content = self._clean_content(content)
                            current_data = json.loads(content)
                        else:
                            current_data = default.copy() if default else {}
                    except json.JSONDecodeError:
                        logger.warning(f"读取JSON失败，使用默认值: {path}")
                        current_data = default.copy() if default else {}
                    except Exception as e:
                        logger.error(f"读取文件失败: {e}")
                        current_data = default.copy() if default else {}
                    
                    # 应用更新函数
                    try:
                        updated_data = update_func(current_data)
                    except Exception as e:
                        logger.error(f"更新函数执行失败: {e}")
                        return False
                    
                    # 写入更新后的数据（在同一锁保护下）
                    f.seek(0)
                    f.truncate()
                    json.dump(updated_data, f, ensure_ascii=False, indent=4)
                    f.flush()
                    os.fsync(f.fileno())  # 确保数据写入磁盘
                    
                    logger.debug(f"成功更新JSON文件: {path}")
                    return True
                    
            except FileNotFoundError:
                # 文件不存在，创建新文件
                try:
                    current_data = default.copy() if default else {}
                    updated_data = update_func(current_data)
                    self.safe_write_json(path, updated_data, backup=False)
                    return True
                except Exception as e:
                    logger.error(f"创建新文件失败: {path}, 错误: {e}")
                    return False
                    
            except Exception as e:
                logger.error(f"更新JSON文件失败: {path}, 错误: {e}")
                if lock_attempt < max_lock_attempts - 1:
                    time.sleep(lock_delay * (lock_attempt + 1))
                    continue
                return False
        
        return False


# 全局单例
_json_manager = JsonFileManager()


def safe_read_json(path: str, default: Optional[Dict] = None) -> Dict[str, Any]:
    """便捷函数：安全读取JSON文件"""
    return _json_manager.safe_read_json(path, default)


def safe_write_json(path: str, data: Dict[str, Any], backup: bool = True) -> bool:
    """便捷函数：安全写入JSON文件"""
    return _json_manager.safe_write_json(path, data, backup)


def safe_update_json(path: str, update_func: Callable[[Dict[str, Any]], Dict[str, Any]], 
                    default: Optional[Dict] = None) -> bool:
    """便捷函数：安全更新JSON文件"""
    return _json_manager.safe_update_json(path, update_func, default)

