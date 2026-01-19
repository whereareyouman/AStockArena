import { useState, useEffect } from 'react';
import { Newspaper, ExternalLink, Clock } from 'lucide-react';

interface NewsItem {
  title: string;
  content: string;
  publish_time: string;
  symbol: string;
  source: string;
  url: string;
}

export function LiveNewsPanel() {
  const [news, setNews] = useState<NewsItem[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchNews = async () => {
      try {
        // Fetch news for all stocks in the portfolio
        const symbols = 'SH688008,SH688111,SH688009,SH688981,SH688256,SH688271,SH688047,SH688617,SH688303,SH688180';
        const res = await fetch(`http://localhost:8000/api/live/news?limit=15&symbols=${symbols}`);
        if (!res.ok) throw new Error('failed');
        const data = await res.json();
        setNews(data.news || []);
      } catch (err) {
        console.warn('Failed to fetch news:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchNews();
    const interval = setInterval(fetchNews, 5 * 60_000); // Refresh every 5 minutes
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-32">
        <div className="text-gray-400 text-sm">加载新闻中...</div>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {news.length === 0 ? (
        <div className="text-center text-gray-400 text-sm py-8">
          <Newspaper className="w-8 h-8 mx-auto mb-2 opacity-50" />
          暂无相关新闻
        </div>
      ) : (
        news.map((item, idx) => (
          <div
            key={idx}
            className="glass-card rounded-lg p-3 hover:bg-gray-800/50 transition-colors"
          >
            <div className="flex items-start gap-2 mb-2">
              <Newspaper className="w-4 h-4 text-blue-400 mt-1 flex-shrink-0" />
              <div className="flex-1 min-w-0">
                <h3 className="text-white text-sm font-medium line-clamp-2 mb-1">
                  {item.title}
                </h3>
                {item.content && (
                  <p className="text-gray-400 text-xs line-clamp-2 mb-2">
                    {item.content}
                  </p>
                )}
              </div>
            </div>

            <div className="flex items-center justify-between text-xs text-gray-500">
              <div className="flex items-center gap-2">
                {item.symbol && (
                  <span className="px-2 py-0.5 bg-blue-500/20 text-blue-400 rounded">
                    {item.symbol}
                  </span>
                )}
                <span>{item.source}</span>
              </div>
              <div className="flex items-center gap-2">
                {item.publish_time && (
                  <div className="flex items-center gap-1">
                    <Clock className="w-3 h-3" />
                    {new Date(item.publish_time).toLocaleString('zh-CN', {
                      month: 'numeric',
                      day: 'numeric',
                      hour: '2-digit',
                      minute: '2-digit',
                    })}
                  </div>
                )}
                {item.url && (
                  <a
                    href={item.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-blue-400 hover:text-blue-300"
                  >
                    <ExternalLink className="w-3 h-3" />
                  </a>
                )}
              </div>
            </div>
          </div>
        ))
      )}
    </div>
  );
}
