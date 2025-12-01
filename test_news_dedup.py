#!/usr/bin/env python3
"""
测试新闻去重功能
"""
import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from tools.news_deduplicator import deduplicate_news_by_embedding

def main():
    print("="*80)
    print("新闻去重功能测试")
    print("="*80)
    print()
    
    # 测试数据：模拟科创板新闻，包含一些相似的标题
    test_news = [
        {
            "title": "科创板平均股价39.44元，8股股价超300元",
            "symbol": "SH688008",
            "publish_time": "2025-11-28 17:59:36"
        },
        {
            "title": "科创板平均股价39.44元 8股股价超300元",  # 与第1条几乎相同
            "symbol": "SH688111",
            "publish_time": "2025-11-28 17:59:36"
        },
        {
            "title": "深沪北百元股数量达153只，电子行业占比最高",
            "symbol": "SH688008",
            "publish_time": "2025-11-28 18:00:10"
        },
        {
            "title": "中芯国际：终止出售中芯宁波股权",
            "symbol": "SH688981",
            "publish_time": "2025-11-28 18:31:20"
        },
        {
            "title": "中芯国际终止出售中芯宁波14.832%股权，交易各方未达一致",  # 与第4条相似
            "symbol": "SH688981",
            "publish_time": "2025-11-28 18:43:00"
        },
        {
            "title": "寒武纪：选举陈天石为董事长",
            "symbol": "SH688256",
            "publish_time": "2025-11-28 18:21:52"
        },
        {
            "title": "寒武纪：选举陈天石为公司第三届董事会董事长",  # 与第6条相似
            "symbol": "SH688256",
            "publish_time": "2025-11-28 18:09:27"
        },
        {
            "title": "科创板晚报|超卓航科实控人拟变更为湖北省国资委 中芯国际终止出售中芯宁波股权",
            "symbol": "SH688981",
            "publish_time": "2025-11-28 21:04:40"
        }
    ]
    
    print(f"📋 原始新闻列表 ({len(test_news)} 条):")
    print("-" * 80)
    for i, news in enumerate(test_news, 1):
        print(f"{i:2d}. [{news['symbol']}] {news['title'][:60]}...")
    
    print("\n" + "="*80)
    print("开始去重（相似度阈值=0.85）...")
    print("="*80 + "\n")
    
    # 执行去重
    try:
        deduplicated = deduplicate_news_by_embedding(
            test_news,
            similarity_threshold=0.85,
            field_to_compare='title'
        )
        
        print(f"\n✅ 去重后的新闻列表 ({len(deduplicated)} 条):")
        print("-" * 80)
        for i, news in enumerate(deduplicated, 1):
            print(f"{i:2d}. [{news['symbol']}] {news['title'][:60]}...")
        
        print("\n" + "="*80)
        print(f"去重统计:")
        print(f"  原始新闻数: {len(test_news)}")
        print(f"  去重后新闻数: {len(deduplicated)}")
        print(f"  移除重复数: {len(test_news) - len(deduplicated)}")
        print("="*80)
        
    except ImportError as e:
        print(f"\n❌ 错误: {e}")
        print("\n💡 请先安装依赖:")
        print("   pip install sentence-transformers")
        return 1
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n✅ 测试完成!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
