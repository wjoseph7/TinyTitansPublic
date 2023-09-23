import numpy as np

def get_adjusted_close(ticker: str, date: str) -> float:
    """
    Summary:
        Uses polygon rest API to get adjusted close for ticker on given date
    Args:
        ticker (str): ticker we're interested in
        date (str): date in YYYY-MM-DD format
    Returns:
        float: adjusted close price of ticker on given date
    """
    split_adj = 'true'
    request_url = f"https://api.polygon.io/v1/open-close/{ticker}/{date}" + \
        f"?adjusted={split_adj}&apiKey={api_key}"
    try:
        response = requests.get(request_url)
        if response.status_code == 200:
            response_dict = response.json()
            last_price = response_dict['close']
            return last_price
        else:
            return np.nan
    except:
        return np.nan