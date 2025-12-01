#!/usr/bin/env python3
"""
辅助脚本：使用环境变量中的账号信息登录 TinySoft 并立即登出，
用于释放可能遗留的远程会话。
"""

import os
import sys

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None


def main() -> int:
    try:
        import pyTSL as ts
    except Exception as e:
        print(f"无法导入 pyTSL: {e}")
        return 1

    if load_dotenv:
        load_dotenv()

    username = os.getenv("TSL_USERNAME") or os.getenv("TSL_USER")
    password = os.getenv("TSL_PASSWORD") or os.getenv("TSL_PASS")
    server = os.getenv("TSL_SERVER", "tsl.tinysoft.com.cn")
    port = int(os.getenv("TSL_PORT", "443"))

    if not username or not password:
        print("缺少环境变量 TSL_USERNAME / TSL_PASSWORD，无法登出。")
        return 1

    print(f"使用账号 {username} 登录 {server}:{port}，尝试释放远程会话...")

    try:
        client = ts.Client(username, password, server, port)
    except Exception as e:
        print(f"初始化 TinySoft Client 失败: {e}")
        return 1

    try:
        ok = client.login()
        if ok != 1:
            last_err = getattr(client, "last_error", lambda: "unknown")()
            err_text = str(last_err)
            print(f"登录失败：{err_text}")
            if "Relogin refused" in err_text:
                print("检测到远程已有会话，正在尝试执行 logout() 释放...")
            try:
                client.logout()
                print("已调用 logout() 释放会话。")
            except Exception as logout_err:
                print(f"执行 logout() 时发生异常: {logout_err}")
            return 1

        print("TinySoft 登录成功，正在执行 logout() ...")
        client.logout()
        print("TinySoft 登出完成。")
        return 0
    except Exception as e:
        print(f"TinySoft 操作异常: {e}")
        try:
            client.logout()
            print("异常后已尝试执行 logout()。")
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    sys.exit(main())

