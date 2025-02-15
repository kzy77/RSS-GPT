const CORS_PROXY = 'https://corsproxy.io/?';
const RSS_URL = 'https://freshrss.chunkj.us.kg/i/?a=rss&user=chunkj&token=chunkj&hours=24';

async function fetchRSSFeeds() {
  try {
    const response = await fetch(CORS_PROXY + encodeURIComponent(RSS_URL));
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    const data = await response.text();
    // ...处理响应数据的代码...
  } catch (error) {
    console.error('获取RSS源时出错:', error);
  }
}
