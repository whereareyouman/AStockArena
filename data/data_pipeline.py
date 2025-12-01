import pyTSL as ts
from datetime import datetime, timedelta  
import json
import pandas as pd 
import time

# 股票列表（科创板代表性股票）
DEFAULT_STOCK_SYMBOLS = [
    "SH688008",  # 澜起科技
    "SH688111",  # 金山办公
    "SH688009",  # 中国通号
    "SH688981",  # 中芯国际
    "SH688256",  # 寒武纪
    "SH688271",  # 联影医疗
    "SH688047",  # 龙芯中科
    "SH688617",  # 惠泰医疗
    "SH688303",  # 大全能源
    "SH688180",  # 君实生物
]

# 登录函数
def login_tinysoft():
    username = "liuzeyu" 
    password = "liuzeyu"  
    server = "tsl.tinysoft.com.cn"
    port = 443

    try:
        t0 = time.time()
        c = ts.Client(username, password, server, port)
        ok = c.login()
        dt = time.time() - t0
        if ok != 1:
            print(f"登录失败: {c.last_error()}  (耗时 {dt:.2f}s)")
            return None
        print(f"登录成功 (耗时 {dt:.2f}s)")
        return c
    except Exception as e:
        print(f"登录异常: {e}")
        return None


# 获取股票列表
def get_top_stocks(top_n=10):
    return DEFAULT_STOCK_SYMBOLS[:top_n]

# Pipeline函数：取最近 ndays 天的数据（日线 + 小时线 + 指标）
# 遵循天软"一次登录，多次交互"原则，接受已登录的客户端对象
def get_stock_data(c, stock, ndays=60):  # 默认60天，确保有足够数据计算技术指标
    """
    获取单只股票的数据（日线 + 小时线 + 指标）
    
    Args:
        c: 已登录的 TinySoft 客户端对象
        stock: 股票代码
        ndays: 回溯天数，默认60天
    
    Returns:
        dict: 包含日线行情、日线指标、小时线行情、小时线指标的字典
    """
    if c is None:
        return None
    
    now = datetime.now()
    begin_time = now - timedelta(days=ndays)  # 最近 ndays 天开始
    end_time = now
    
    data = {}
    
    # 取日线行情
    r_day = c.query(stock=stock, begin_time=begin_time, end_time=end_time, cycle='日线', fields='date, close, vol, amount, buy1')
    if r_day.error() == 0:
        df = r_day.dataframe()
        df['date'] = df['date'].dt.strftime('%Y-%m-%d')
        data['日线行情'] = df.to_dict(orient='records')
    
    # 取日线指标
    end_date = now.strftime('%Y%m%d')
    tsl_day = f"""
    SetSysParam(pn_stock(),'{stock}');
    SetSysParam(pn_date(),{end_date}T);
    setsysparam(pn_cycle(),cy_day());
    setsysparam(pn_rate(),0);
    setsysparam(pn_Nday(),{ndays});
    V:=KDJ_f(9,3,3,1);
    B:=boll_f(20,1);
    D:=Nday2("Date",datetostr(sp_time()));
    Return D|V[0]|V[1]|V[2]|B[0]|B[1]|B[2]|B[3];
    """
    r_day_ind = c.exec(tsl_day)
    if r_day_ind.error() == 0:
        data['日线指标'] = r_day_ind.value()  # table 格式数组
    
    # 取小时线行情（每小时数据）
    r_hour = c.query(stock=stock, begin_time=begin_time, end_time=end_time, cycle='60分钟线', fields='date, close, vol, amount, buy1')
    if r_hour.error() == 0:
        df = r_hour.dataframe()
        df['date'] = df['date'].dt.strftime('%Y-%m-%d %H:%M:%S')
        data['小时线行情'] = df.to_dict(orient='records')
    
    # 取小时线指标
    tsl_hour = f"""
    SetSysParam(pn_stock(),'{stock}');
    SetSysParam(pn_date(),{end_date}T);
    setsysparam(pn_cycle(),cy_60m());
    setsysparam(pn_rate(),0);
    setsysparam(pn_Nday(),{ndays});
    V:=KDJ_f(9,3,3,1);
    B:=boll_f(20,1);
    D:=Nday2("Date",datetostr(sp_time()));
    Return D|V[0]|V[1]|V[2]|B[0]|B[1]|B[2]|B[3];
    """
    r_hour_ind = c.exec(tsl_hour)
    if r_hour_ind.error() == 0:
        data['小时线指标'] = r_hour_ind.value()
    
    return data

# 使用示例（取 top 10 股票的数据）
# 遵循天软"一次登录，多次交互"原则：一次登录，处理所有股票，最后统一退出
if __name__ == "__main__":
    top_stocks = get_top_stocks()
    all_data = {}
    
    # 一次登录
    c = login_tinysoft()
    if not c:
        print("❌ 登录失败，无法获取数据")
        exit(1)
    
    try:
        # 多次交互：循环处理所有股票
        for stock in top_stocks:
            print(f"正在获取 {stock} 的数据...")
            data = get_stock_data(c, stock, ndays=60)
            if data:
                all_data[stock] = data
                print(f"✅ {stock} 数据获取成功")
                # 每次查询后稍作延迟，避免请求过于频繁
                time.sleep(0.5)
            else:
                print(f"❌ {stock} 数据获取失败")
    finally:
        # 最后统一退出：logout 后删除对象并等待，确保注销完成
        try:
            c.logout()
            del c
            time.sleep(1.5)  # sleep 1-2秒，保证本次登录注销完成
            print("✅ TinySoft 会话已安全关闭")
        except Exception as e:
            print(f"⚠️ 退出登录时出现异常: {e}")
    
    if all_data:
        import os
        import sys
        # 确保输出目录存在
        output_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(output_dir, 'ai_stock_data.json')
        
        # 使用JsonFileManager安全写入
        sys.path.insert(0, os.path.dirname(output_dir))
        from tools.json_file_manager import safe_write_json
        
        if safe_write_json(output_path, all_data, backup=True):
            print(f"✅ 数据已保存到 {output_path}（{len(all_data)} 只股票）")
        else:
            print(f"❌ 保存数据失败: {output_path}")
    else:
        print("❌ 获取数据失败")