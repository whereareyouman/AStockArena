from __future__ import annotations

from typing import Any


TUSHARE_TOKEN = "4BNXJQbVdwyhYy5jNAqz4zEIl8RPtH18"
TUSHARE_HTTP_URL = "https://tsp.jingjingtech.com"


def get_tushare_pro() -> Any:
    """Return a Tushare Pro client using the project-wide token and proxy."""
    import tushare as ts

    ts.set_token(TUSHARE_TOKEN)
    pro = ts.pro_api()
    pro._DataApi__http_url = TUSHARE_HTTP_URL
    return pro


def get_tushare_module():
    """Return the tushare module after setting the project-wide token."""
    import tushare as ts

    ts.set_token(TUSHARE_TOKEN)
    return ts
