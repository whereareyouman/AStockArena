# LLM决策控制机制说明

## 📋 当前运行模式

### 手动触发模式 (Current Implementation)

**工作流程**:
1. 用户在Web界面点击 **"Start Trading"** 按钮
2. 前端发送请求到后端 `/api/job/start`
3. 后端启动 `main.py` 子进程作为后台任务
4. `main.py` 读取配置文件 (`configs/default_config.json`):
   - `init_date`: 起始日期 (例如: 2025-10-30)
   - `end_date`: 结束日期 (例如: 2025-11-08)
5. 遍历日期范围内的每个交易日:
   - 检查是否为交易日 (跳过周末/节假日)
   - 在每个交易日的3个固定时点触发LLM决策:
     - **09:30** - 开盘观察
     - **11:30** - 午间部署
     - **14:00** - 午后调整
6. 每次决策时:
   - 获取最新行情数据 (TinySoft)
   - 获取相关新闻 (AKShare)
   - 构建prompt发送给LLM (Gemini 2.5 Flash)
   - LLM返回决策: `buy` / `sell` / `no_trade`
   - 更新投资组合状态
   - 写入 `position.jsonl`
   - 记录决策日志到 `log/{date}/session_*.jsonl`
7. 处理完所有日期后结束，前端显示"完成"

**代码位置**:
- 前端触发: `Tradingsimulation/src/components/sci/TradingControl.tsx`
- 后端API: `api_server.py` - `/api/job/start` 和 `/api/job/{job_id}`
- 主逻辑: `main.py`
- AI Agent: `agent/base_agent/base_agent.py`

### 当前默认股票池

> 澜起科技(SH688008)、金山办公(SH688111)、中国通号(SH688009)、中芯国际(SH688981)、寒武纪(SH688256)、联影医疗(SH688271)、龙芯中科(SH688047)、惠泰医疗(SH688617)、大全能源(SH688303)、君实生物(SH688180)。

---

## 🎯 目标运行模式

### 自动化定时决策 (Desired Implementation)

**理想工作流程**:
1. **服务常驻运行**: `main.py` 或调度器持续在后台运行
2. **定时触发**: 每小时自动检查是否需要做决策
3. **智能判断**:
   - 检查当前时间
   - 判断是否为交易时段 (09:30-15:00)
   - 判断今天是否为交易日
4. **自动执行决策**: 到达触发时点自动调用LLM并执行交易
5. **持续监控**: 24/7运行，自动处理每个交易日

---

## 🔄 实现方案对比

### 方案1: Cron定时任务 (推荐)

**优点**:
- 系统级可靠性高
- 不依赖进程持续运行
- 易于监控和重启

**实现步骤**:

#### 1.1 创建简化的runner脚本

```bash
# /Users/fangdoudou/Desktop/urop25-26/AI-Trader_11_8/run_hourly_decision.sh

#!/bin/bash
set -e

cd "$(dirname "$0")"

# 加载环境变量
source env.sh

# 设置今天为交易日期
export INIT_DATE=$(date +%Y-%m-%d)
export END_DATE=$(date +%Y-%m-%d)

# 运行单次决策
python3 main.py --mode=single-decision

# 记录执行日志
echo "[$(date)] Hourly decision executed" >> logs/hourly_decision.log
```

#### 1.2 修改main.py支持单次决策模式

```python
# main.py 新增参数

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['full', 'single-decision'], default='full',
                    help='full: 处理整个日期范围; single-decision: 仅执行当前时点的一次决策')
args = parser.parse_args()

if args.mode == 'single-decision':
    # 只执行当前时刻的决策
    current_time = datetime.now()
    if not is_trading_day(current_time.date()):
        print("今天不是交易日")
        sys.exit(0)
    
    hour = current_time.hour
    minute = current_time.minute
    
    # 判断是否在决策时点附近 (允许±5分钟容差)
    decision_times = [(9, 30), (11, 30), (14, 0)]
    should_decide = False
    for dh, dm in decision_times:
        if abs((hour * 60 + minute) - (dh * 60 + dm)) <= 5:
            should_decide = True
            break
    
    if not should_decide:
        print(f"当前时间 {hour}:{minute:02d} 不在决策时点")
        sys.exit(0)
    
    # 执行决策
    execute_single_decision(agent, current_time)
    sys.exit(0)
```

#### 1.3 设置crontab

```bash
# 编辑crontab
crontab -e

# 添加以下行 (在三个时点前几分钟唤醒)
25 9 * * 1-5 /Users/fangdoudou/Desktop/urop25-26/AI-Trader_11_8/run_hourly_decision.sh
25 11 * * 1-5 /Users/fangdoudou/Desktop/urop25-26/AI-Trader_11_8/run_hourly_decision.sh
55 13 * * 1-5 /Users/fangdoudou/Desktop/urop25-26/AI-Trader_11_8/run_hourly_decision.sh

# 或者每小时执行，脚本内部判断
0 9-15 * * 1-5 /Users/fangdoudou/Desktop/urop25-26/AI-Trader_11_8/run_hourly_decision.sh
```

**解释**:
- `30 9-15 * * 1-5`: 周一到周五，9点到15点的每个小时的30分
- `1-5`: 周一(1)到周五(5)
- 或者简化为每小时执行，脚本内判断是否需要决策

---

### 方案2: systemd定时服务 (Linux)

**优点**:
- 更现代的Linux服务管理
- 自动重启失败任务
- 详细的日志记录

**实现步骤**:

#### 2.1 创建systemd timer

```ini
# /etc/systemd/system/ai-trader-hourly.timer

[Unit]
Description=AI Trader Hourly Decision Timer
Requires=ai-trader-hourly.service

[Timer]
OnCalendar=Mon..Fri 09:30:00
OnCalendar=Mon..Fri 11:30:00
OnCalendar=Mon..Fri 14:00:00
Persistent=true

[Install]
WantedBy=timers.target
```

#### 2.2 创建systemd service

```ini
# /etc/systemd/system/ai-trader-hourly.service

[Unit]
Description=AI Trader Hourly Decision Service

[Service]
Type=oneshot
User=fangdoudou
WorkingDirectory=/Users/fangdoudou/Desktop/urop25-26/AI-Trader_11_8
ExecStart=/usr/bin/bash run_hourly_decision.sh
StandardOutput=journal
StandardError=journal
```

#### 2.3 启用timer

```bash
sudo systemctl daemon-reload
sudo systemctl enable ai-trader-hourly.timer
sudo systemctl start ai-trader-hourly.timer

# 查看状态
sudo systemctl status ai-trader-hourly.timer
sudo systemctl list-timers --all
```

---

### 方案3: Python调度器 (APScheduler)

**优点**:
- 纯Python实现，跨平台
- 灵活的调度规则
- 可集成到现有后端

**实现步骤**:

#### 3.1 安装APScheduler

```bash
pip install apscheduler
```

#### 3.2 创建调度服务

```python
# scheduler_service.py

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from datetime import datetime
import subprocess
import os

def is_trading_day():
    """检查今天是否为交易日 (简化版)"""
    today = datetime.now()
    # 周末不交易
    if today.weekday() >= 5:
        return False
    # 可添加节假日判断
    return True

def execute_decision():
    """执行一次决策"""
    if not is_trading_day():
        print(f"{datetime.now()}: 今天不是交易日，跳过")
        return
    
    print(f"{datetime.now()}: 执行LLM决策...")
    try:
        # 设置环境变量
        env = os.environ.copy()
        env['INIT_DATE'] = datetime.now().strftime('%Y-%m-%d')
        env['END_DATE'] = datetime.now().strftime('%Y-%m-%d')
        
        # 运行main.py
        result = subprocess.run(
            ['python3', 'main.py', '--mode=single-decision'],
            env=env,
            capture_output=True,
            text=True,
            timeout=300  # 5分钟超时
        )
        
        print(f"决策完成，退出码: {result.returncode}")
        if result.stdout:
            print(f"输出: {result.stdout}")
        if result.stderr:
            print(f"错误: {result.stderr}")
    
    except Exception as e:
        print(f"决策执行失败: {e}")

def main():
    scheduler = BlockingScheduler()
    
    # 添加定时任务 - 交易日的指定时点
    decision_times = [
        ('09:30', 'cron', {'day_of_week': 'mon-fri', 'hour': 9, 'minute': 30}),
        ('11:30', 'cron', {'day_of_week': 'mon-fri', 'hour': 11, 'minute': 30}),
        ('14:00', 'cron', {'day_of_week': 'mon-fri', 'hour': 14, 'minute': 0}),
    ]
    
    for name, trigger_type, trigger_args in decision_times:
        scheduler.add_job(
            execute_decision,
            trigger=CronTrigger(**trigger_args),
            id=f'decision_{name}',
            name=f'LLM决策 {name}',
            misfire_grace_time=300  # 错过执行时间后5分钟内仍可执行
        )
    
    print("调度器已启动，等待执行时点...")
    print("已注册的任务:")
    for job in scheduler.get_jobs():
        print(f"  - {job.name} (下次执行: {job.next_run_time})")
    
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("\n调度器已停止")

if __name__ == '__main__':
    main()
```

#### 3.3 使用supervisor保持运行

```ini
# /etc/supervisor/conf.d/ai-trader-scheduler.conf

[program:ai-trader-scheduler]
directory=/Users/fangdoudou/Desktop/urop25-26/AI-Trader_11_8
command=/path/to/venv/bin/python scheduler_service.py
user=fangdoudou
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/Users/fangdoudou/Desktop/urop25-26/AI-Trader_11_8/logs/scheduler.log
```

---

### 方案4: 集成到FastAPI后端

**优点**:
- 与现有后端统一管理
- Web界面可查看调度状态
- 易于调试和监控

**实现步骤**:

#### 4.1 修改api_server.py

```python
# api_server.py 添加

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

# 全局调度器
scheduler = BackgroundScheduler()

def trigger_llm_decision():
    """后台任务: 触发LLM决策"""
    # 复用现有的job启动逻辑
    job_id = str(uuid.uuid4())
    log_file = LOG_DIR / f"job_{job_id}.log"
    
    env = os.environ.copy()
    env['INIT_DATE'] = datetime.now().strftime('%Y-%m-%d')
    env['END_DATE'] = datetime.now().strftime('%Y-%m-%d')
    
    proc = subprocess.Popen(
        [sys.executable, "main.py", "--mode=single-decision"],
        stdout=open(log_file, "w"),
        stderr=subprocess.STDOUT,
        env=env
    )
    
    JOBS[job_id] = {
        "id": job_id,
        "status": "running",
        "started_at": datetime.now().isoformat(),
        "proc": proc,
        "log_file": str(log_file)
    }
    
    print(f"自动决策任务已启动: {job_id}")

@app.on_event("startup")
async def startup_event():
    """应用启动时初始化调度器"""
    decision_times = [
        {'day_of_week': 'mon-fri', 'hour': 9, 'minute': 30},
        {'day_of_week': 'mon-fri', 'hour': 10, 'minute': 30},
        {'day_of_week': 'mon-fri', 'hour': 11, 'minute': 0},
        {'day_of_week': 'mon-fri', 'hour': 13, 'minute': 0},
        {'day_of_week': 'mon-fri', 'hour': 14, 'minute': 0},
        {'day_of_week': 'mon-fri', 'hour': 15, 'minute': 0},
    ]
    
    for i, trigger_args in enumerate(decision_times):
        scheduler.add_job(
            trigger_llm_decision,
            trigger=CronTrigger(**trigger_args),
            id=f'auto_decision_{i}',
            misfire_grace_time=300
        )
    
    scheduler.start()
    print("自动决策调度器已启动")

@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时停止调度器"""
    scheduler.shutdown()
    print("调度器已停止")

@app.get("/api/scheduler/status")
async def scheduler_status():
    """查看调度器状态"""
    jobs = []
    for job in scheduler.get_jobs():
        jobs.append({
            "id": job.id,
            "name": job.name,
            "next_run": job.next_run_time.isoformat() if job.next_run_time else None
        })
    return {"running": scheduler.running, "jobs": jobs}
```

#### 4.2 在前端显示调度状态

```tsx
// 新增组件: SchedulerStatus.tsx

export function SchedulerStatus() {
  const [status, setStatus] = useState<any>(null);
  
  useEffect(() => {
    const fetch = async () => {
      const res = await fetch('http://localhost:8000/api/scheduler/status');
      const data = await res.json();
      setStatus(data);
    };
    fetch();
    const interval = setInterval(fetch, 30_000);
    return () => clearInterval(interval);
  }, []);
  
  return (
    <div className="glass-card p-4">
      <h3 className="text-white mb-2">自动决策调度器</h3>
      <div className="text-sm">
        <div>状态: {status?.running ? '运行中' : '已停止'}</div>
        <div className="mt-2">下次执行:</div>
        {status?.jobs?.map((job: any) => (
          <div key={job.id} className="text-gray-400">
            {new Date(job.next_run).toLocaleString('zh-CN')}
          </div>
        ))}
      </div>
    </div>
  );
}
```

---

## ⚙️ 配置建议

### 环境变量

```bash
# env.sh 添加

# 决策模式
export DECISION_MODE="auto"  # auto: 自动定时; manual: 手动触发

# 决策时点 (逗号分隔，格式 HH:MM)
export DECISION_TIMES="09:30,11:30,14:00"

# 是否启用自动决策
export ENABLE_AUTO_DECISION="true"
```

### 日志配置

```python
# 在main.py或scheduler中配置日志

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/auto_decision.log'),
        logging.StreamHandler()
    ]
)
```

---

## 📊 监控和告警

### 健康检查

```python
# health_check.py

import requests
from datetime import datetime

def check_scheduler():
    """检查调度器是否正常运行"""
    try:
        res = requests.get('http://localhost:8000/api/scheduler/status', timeout=5)
        if res.status_code == 200:
            data = res.json()
            if data.get('running'):
                return "OK"
        return "调度器未运行"
    except Exception as e:
        return f"检查失败: {e}"

def check_last_decision():
    """检查最后一次决策时间"""
    try:
        res = requests.get('http://localhost:8000/api/live/recent-decisions?limit=1')
        if res.status_code == 200:
            data = res.json()
            if data.get('decisions'):
                last_time = data['decisions'][0].get('time')
                # 检查是否在预期时间内
                return f"最后决策: {last_time}"
        return "无决策记录"
    except Exception as e:
        return f"检查失败: {e}"

if __name__ == '__main__':
    print(f"[{datetime.now()}] 系统健康检查")
    print(f"  调度器: {check_scheduler()}")
    print(f"  决策: {check_last_decision()}")
```

### 告警通知 (可选)

```python
# 添加钉钉/企业微信/邮件通知

def send_alert(message: str):
    """发送告警消息"""
    # 钉钉机器人
    webhook_url = "https://oapi.dingtalk.com/robot/send?access_token=YOUR_TOKEN"
    requests.post(webhook_url, json={
        "msgtype": "text",
        "text": {"content": f"AI Trader 告警: {message}"}
    })

# 在决策失败时调用
try:
    execute_decision()
except Exception as e:
    send_alert(f"决策执行失败: {e}")
```

---

## 🎯 推荐实施路径

### 阶段1: 测试单次决策 (1天)
1. 修改`main.py`添加`--mode=single-decision`参数
2. 手动运行测试: `python main.py --mode=single-decision`
3. 验证能正确执行单次决策并写入数据

### 阶段2: Cron定时任务 (1天)
1. 创建`run_hourly_decision.sh`脚本
2. 设置crontab每小时执行
3. 观察1个交易日，确认3次决策都正常执行

### 阶段3: 集成到后端 (2天)
1. 在`api_server.py`添加APScheduler
2. 添加`/api/scheduler/status`端点
3. 前端添加调度器状态显示

### 阶段4: 监控和优化 (持续)
1. 添加健康检查脚本
2. 配置日志轮转
3. 添加告警通知
4. 优化决策性能

---

## 📝 注意事项

1. **幂等性**: 确保同一时点多次执行不会重复决策
2. **并发控制**: 避免多个决策任务同时运行
3. **错误恢复**: 决策失败后的重试机制
4. **节假日处理**: 维护交易日历数据
5. **API限流**: LLM API可能有速率限制
6. **数据一致性**: 确保position.jsonl写入的原子性
7. **资源管理**: 定期清理旧日志文件
8. **权限管理**: cron任务需要正确的文件权限

---

## 🔗 相关文档

- **数据存储**: `DATA_STORAGE.md`
- **快速开始**: `QUICKSTART.md`
- **UI改进**: `UI_IMPROVEMENTS_SUMMARY.md`
- **后端API**: `api_server.py` (查看所有端点)

---

**最后更新**: 2025-11-13  
**版本**: v1.0 - LLM Decision Control Documentation
