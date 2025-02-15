import feedparser
import configparser
import os
import httpx
from openai import OpenAI
from jinja2 import Template
from bs4 import BeautifulSoup
import re
import datetime
import requests
from fake_useragent import UserAgent
import glob
import time
from zoneinfo import ZoneInfo  # 使用 zoneinfo 模块处理时区
#from dateutil.parser import parse
from google import genai

def get_cfg(sec, name, default=None):
    value=config.get(sec, name, fallback=default)
    if value:
        return value.strip('"')

config = configparser.ConfigParser()
config.read('config.ini')
secs = config.sections()
# Maxnumber of entries to in a feed.xml file
max_entries = 1000

# OpenAI 配置
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
# 确保 base URL 总是以 /v1 结尾
base_url = os.environ.get('OPENAI_BASE_URL', 'https://api.openai.com')
OPENAI_BASE_URL = base_url if base_url.endswith('/v1') else f"{base_url.rstrip('/')}/v1"
OPENAI_MODEL = os.environ.get('OPENAI_MODEL', 'gpt-3.5-turbo')

# Gemini 配置
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
GEMINI_BASE_URL = os.environ.get('GEMINI_BASE_URL')  # 如果未设置则使用默认
GEMINI_MODEL = os.environ.get('GEMINI_MODEL', 'gemini-2.0-flash-exp')

# 默认提供商
DEFAULT_PROVIDER = os.environ.get('DEFAULT_PROVIDER', 'openai')

# 修改 Gemini 初始化
if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)

# 限流控制
class RateLimiter:
    def __init__(self, max_requests, time_window):
        self.max_requests = max_requests  # 最大请求数
        self.time_window = time_window    # 时间窗口（秒）
        self.requests = []                # 请求时间戳列表

    def wait_if_needed(self):
        now = time.time()
        # 清理过期的请求记录
        self.requests = [req_time for req_time in self.requests 
                        if now - req_time < self.time_window]
        
        if len(self.requests) >= self.max_requests:
            # 需要等待的时间
            sleep_time = self.requests[0] + self.time_window - now
            if sleep_time > 0:
                print(f"Rate limit reached, waiting for {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
            # 清理已过期的请求
            self.requests = self.requests[1:]
        
        self.requests.append(now)

# 创建限流器实例：每分钟最多10次请求
rate_limiter = RateLimiter(max_requests=10, time_window=60)

def gemini_summary(query, language):
    """使用 Gemini 生成摘要"""
    try:
        rate_limiter.wait_if_needed()
        print(f"Gemini Config - Model: {GEMINI_MODEL}")

        if language == "zh":
            prompt = f"请用中文总结这篇文章，先提取出{keyword_length}个关键词，在同一行内输出，然后换行，用中文在{summary_length}字内写一个包含所有要点的总结，按顺序分要点输出，并按照以下格式输出'<br><br>总结:'"
        else:
            prompt = f"Please summarize this article in {language}, first extract {keyword_length} keywords, output in the same line, then line break, write a summary containing all points in {summary_length} words in {language}, output in order by points, and output '<br><br>Summary:'"
        
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=f"{prompt}\n\n{query}",
        )
        return response.text

    except Exception as e:
        raise Exception(f"Gemini summary failed with model {GEMINI_MODEL}: {str(e)}")

# 其他配置
U_NAME = os.environ.get('U_NAME')
deployment_url = f'https://{U_NAME}.github.io/RSS-GPT/'
BASE =get_cfg('cfg', 'BASE')
keyword_length = int(get_cfg('cfg', 'keyword_length'))
summary_length = int(get_cfg('cfg', 'summary_length'))
language = get_cfg('cfg', 'language')

def fetch_feed(url, log_file):
    feed = None
    response = None
    headers = {}
    try:
        ua = UserAgent()
        headers['User-Agent'] = ua.random.strip()
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code == 200:
            feed = feedparser.parse(response.text)
            return {'feed': feed, 'status': 'success'}
        else:
            with open(log_file, 'a') as f:
                f.write(f"Fetch error: {response.status_code}\n")
            return {'feed': None, 'status': response.status_code}
    except requests.RequestException as e:
        with open(log_file, 'a') as f:
            f.write(f"Fetch error: {e}\n")
        return {'feed': None, 'status': 'failed'}

def generate_untitled(entry):
    try: return entry.title
    except: 
        try: return entry.article[:50]
        except: return entry.link


def clean_html(html_content):
    """
    This function is used to clean the HTML content.
    It will remove all the <script>, <style>, <img>, <a>, <video>, <audio>, <iframe>, <input> tags.
    Returns:
        Cleaned text for summarization
    """
    soup = BeautifulSoup(html_content, "html.parser")

    for script in soup.find_all("script"):
        script.decompose()

    for style in soup.find_all("style"):
        style.decompose()

    for img in soup.find_all("img"):
        img.decompose()

    for a in soup.find_all("a"):
        a.decompose()

    for video in soup.find_all("video"):
        video.decompose()

    for audio in soup.find_all("audio"):
        audio.decompose()
    
    for iframe in soup.find_all("iframe"):
        iframe.decompose()
    
    for input in soup.find_all("input"):
        input.decompose()

    return soup.get_text()

def filter_entry(entry, filter_apply, filter_type, filter_rule):
    """
    This function is used to filter the RSS feed.

    Args:
        entry: RSS feed entry
        filter_apply: title, article or link
        filter_type: include or exclude or regex match or regex not match
        filter_rule: regex rule or keyword rule, depends on the filter_type

    Raises:
        Exception: filter_apply not supported
        Exception: filter_type not supported
    """
    if filter_apply == 'title':
        text = entry.title
    elif filter_apply == 'article':
        text = entry.article
    elif filter_apply == 'link':
        text = entry.link
    elif not filter_apply:
        return True
    else:
        raise Exception('filter_apply not supported')

    if filter_type == 'include':
        return re.search(filter_rule, text)
    elif filter_type == 'exclude':
        return not re.search(filter_rule, text)
    elif filter_type == 'regex match':
        return re.search(filter_rule, text)
    elif filter_type == 'regex not match':
        return not re.search(filter_rule, text)
    elif not filter_type:
        return True
    else:
        raise Exception('filter_type not supported')

def read_entry_from_file(sec):
    """
    This function is used to read the RSS feed entries from the feed.xml file.

    Args:
        sec: section name in config.ini
    """
    out_dir = os.path.join(BASE, get_cfg(sec, 'name'))
    try:
        with open(out_dir + '.xml', 'r') as f:
            rss = f.read()
        feed = feedparser.parse(rss)
        return feed.entries
    except:
        return []

def truncate_entries(entries, max_entries):
    if len(entries) > max_entries:
        entries = entries[:max_entries]
    return entries

def gpt_summary(query, model, language):
    """使用 OpenAI GPT 生成摘要"""
    try:
        rate_limiter.wait_if_needed()
        print(f"OpenAI Config - Base URL: {OPENAI_BASE_URL}, Model: {model}")
        if language == "zh":
            messages = [
                {"role": "user", "content": query},
                {"role": "assistant", "content": f"请用中文总结这篇文章，先提取出{keyword_length}个关键词，在同一行内输出，然后换行，用中文在{summary_length}字内写一个包含所有要点的总结，按顺序分要点输出，并按照以下格式输出'<br><br>总结:'"}
            ]
        else:
            messages = [
                {"role": "user", "content": query},
                {"role": "assistant", "content": f"Please summarize this article in {language}, first extract {keyword_length} keywords, output in the same line, then line break, write a summary containing all points in {summary_length} words in {language}, output in order by points, and output '<br><br>Summary:'"}
            ]

        client = OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
        )

        completion = client.chat.completions.create(
            model=model,
            messages=messages,
        )
        return completion.choices[0].message.content
    except Exception as e:
        raise Exception(f"GPT summary failed with model {model}: {str(e)}")

def output(sec, language):
    """ output
    This function is used to output the summary of the RSS feed.

    Args:
        sec: section name in config.ini

    Raises:
        Exception: filter_apply, type, rule must be set together in config.ini
    """
    log_file = os.path.join(BASE, get_cfg(sec, 'name') + '.log')
    out_dir = os.path.join(BASE, get_cfg(sec, 'name'))
    # read rss_url as a list separated by comma
    rss_urls = get_cfg(sec, 'url')
    rss_urls = rss_urls.split(',')

    # RSS feed filter apply, filter title, article or link, summarize title, article or link
    filter_apply = get_cfg(sec, 'filter_apply')

    # RSS feed filter type, include or exclude or regex match or regex not match
    filter_type = get_cfg(sec, 'filter_type')

    # Regex rule or keyword rule, depends on the filter_type
    filter_rule = get_cfg(sec, 'filter_rule')

    # filter_apply, type, rule must be set together
    if filter_apply and filter_type and filter_rule:
        pass
    elif not filter_apply and not filter_type and not filter_rule:
        pass
    else:
        raise Exception('filter_apply, type, rule must be set together')

    # Max number of items to summarize
    try:
        max_items = int(get_cfg(sec, 'max_items', '10'))  # 默认为10
    except ValueError:
        max_items = 10
        print(f"Warning: Invalid max_items value for {get_cfg(sec, 'name')}, using default: 10")

    cnt = 0
    existing_entries = read_entry_from_file(sec)
    with open(log_file, 'a') as f:
        f.write('------------------------------------------------------\n')
        f.write(f'Started: {datetime.datetime.now()}\n')
        f.write(f'Existing_entries: {len(existing_entries)}\n')
    existing_entries = truncate_entries(existing_entries, max_entries=max_entries)
    # Be careful when the deleted ones are still in the feed, in that case, you will mess up the order of the entries.
    # Truncating old entries is for limiting the file size, 1000 is a safe number to avoid messing up the order.
    append_entries = []

    for rss_url in rss_urls:
        with open(log_file, 'a') as f:
            f.write(f"Fetching from {rss_url}\n")
            print(f"Fetching from {rss_url}")
        feed = fetch_feed(rss_url, log_file)['feed']
        if not feed:
            with open(log_file, 'a') as f:
                f.write(f"Fetch failed from {rss_url}\n")
            continue
        for entry in feed.entries:
            if cnt > max_entries:
                with open(log_file, 'a') as f:
                    f.write(f"Skip from: [{entry.title}]({entry.link})\n")
                break

            if entry.link.find('#replay') and entry.link.find('v2ex'):
                entry.link = entry.link.split('#')[0]

            if entry.link in [x.link for x in existing_entries]:
                continue

            if entry.link in [x.link for x in append_entries]:
                continue

            entry.title = generate_untitled(entry)

            try:
                entry.article = entry.content[0].value
            except:
                try: entry.article = entry.description
                except: entry.article = entry.title

            cleaned_article = clean_html(entry.article)

            if not filter_entry(entry, filter_apply, filter_type, filter_rule):
                with open(log_file, 'a') as f:
                    f.write(f"Filter: [{entry.title}]({entry.link})\n")
                continue


#            # format to Thu, 27 Jul 2023 13:13:42 +0000
#            if 'updated' in entry:
#                entry.updated = parse(entry.updated).strftime('%a, %d %b %Y %H:%M:%S %z')
#            if 'published' in entry:
#                entry.published = parse(entry.published).strftime('%a, %d %b %Y %H:%M:%S %z')

            cnt += 1
            if cnt > max_items:
                entry.summary = None
            else:
                token_length = len(cleaned_article)
                try:
                    provider = get_cfg(sec, 'provider', DEFAULT_PROVIDER)
                    model = get_cfg(sec, 'model', OPENAI_MODEL if provider.lower() == 'openai' else GEMINI_MODEL)
                    # 打印处理信息
                    print(f"Processing '{get_cfg(sec, 'name')}' with {provider}/{model}")
                    
                    if provider.lower() == 'gemini':
                        entry.summary = gemini_summary(cleaned_article, language)
                    else:  # openai
                        entry.summary = gpt_summary(cleaned_article, model, language)
                    
                    with open(log_file, 'a') as f:
                        f.write(f"Token length: {token_length}\n")
                        f.write(f"Summarized using {provider} model: {model}\n")
                except Exception as e:
                    error_msg = f"Summarization failed for '{get_cfg(sec, 'name')}' ({provider}/{model}): {str(e)}"
                    print(error_msg)
                    entry.summary = None
                    with open(log_file, 'a') as f:
                        f.write(f"{error_msg}\n")

            append_entries.append(entry)
            with open(log_file, 'a') as f:
                f.write(f"Append: [{entry.title}]({entry.link})\n")

    with open(log_file, 'a') as f:
        f.write(f'append_entries: {len(append_entries)}\n')

    template = Template(open('template.xml').read())
    
    try:
        rss = template.render(feed=feed, append_entries=append_entries, existing_entries=existing_entries)
        with open(out_dir + '.xml', 'w') as f:
            f.write(rss)
        with open(log_file, 'a') as f:
            f.write(f'Finish: {datetime.datetime.now()}\n')
        return {'feed': feed, 'entries': append_entries, 'section': sec}
    except:
        with open (log_file, 'a') as f:
            f.write(f"error when rendering xml, skip {out_dir}\n")
            print(f"error when rendering xml, skip {out_dir}\n")
        return None

def clean_logs():
    """清理docs目录下的所有log文件"""
    log_files = glob.glob(os.path.join(BASE, "*.log"))
    for log_file in log_files:
        try:
            os.remove(log_file)
            print(f"已删除日志文件: {log_file}")
        except Exception as e:
            print(f"删除日志文件失败 {log_file}: {str(e)}")

def append_readme(readme, links):
    """更新 README 文件"""
    with open(readme, 'r') as f:
        readme_lines = f.readlines()
    while readme_lines[-1].startswith('- ') or readme_lines[-1] == '\n':
        readme_lines = readme_lines[:-1]  # remove 1 line from the end for each feed
    readme_lines.append('\n')
    readme_lines.extend(links)
    with open(readme, 'w') as f:
        f.writelines(readme_lines)

def generate_atom_feed(feeds_data):
    """生成聚合的atom.xml，包含所有源的条目"""
    template = Template(open('template.xml').read())
    
    # 合并所有源的条目
    all_entries = []
    for feed_data in feeds_data:
        if feed_data and 'entries' in feed_data:
            all_entries.extend(feed_data['entries'])
    
    # 按时间排序（如果有更新时间的话）
    try:
        all_entries.sort(key=lambda x: x.updated if hasattr(x, 'updated') else '', reverse=True)
    except:
        pass
    
    # 创建聚合feed的基本信息
    aggregated_feed = {
        'feed': {
            'title': 'RSS-GPT Aggregated Feed',
            'link': deployment_url,
            'updated': datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
            'author': U_NAME,
            'id': deployment_url
        }
    }
    
    # 限制条目数量，避免文件过大
    all_entries = all_entries[:max_entries]
    
    try:
        with open(os.path.join(BASE, 'atom.xml'), 'w', encoding='utf-8') as f:
            rss = template.render(
                feed=aggregated_feed, 
                append_entries=all_entries,
                existing_entries=[]
            )
            f.write(rss)
            print(f"Successfully generated atom.xml with {len(all_entries)} entries")
    except Exception as e:
        print(f"Error generating atom.xml: {str(e)}")

try:
    os.mkdir(BASE)
except:
    pass

# 在处理RSS之前清理日志文件
clean_logs()

# 收集所有feed数据
feeds_data = []
feeds = []
links = []

for x in secs[1:]:  # 跳过[cfg]部分
    feed_data = output(x, language=language)
    if feed_data:
        feeds_data.append(feed_data)
        feeds.append(feed_data['feed'])
        links.append("- "+ get_cfg(x, 'url').replace(',',', ') + " -> " + deployment_url + feed_data['feed']['feed']['title'] + ".xml\n")

# 更新 README
append_readme("README.md", links)
append_readme("README-zh.md", links)

# 生成 index.html
with open(os.path.join(BASE, 'index.html'), 'w') as f:
    template = Template(open('template.html').read())
    # 为每个 feed 添加配置信息
    for feed_data in feeds_data:  # 使用 feeds_data 而不是 feeds
        feed_data['feed']['url'] = get_cfg(feed_data['section'], 'url')
        feed_data['feed']['name'] = get_cfg(feed_data['section'], 'name')
    
    # 使用Asia/Shanghai时区获取当前时间
    shanghai_tz = ZoneInfo("Asia/Shanghai")
    current_time = datetime.datetime.now(shanghai_tz).strftime("%Y-%m-%d %H:%M:%S")
    
    html = template.render(
        update_time=current_time,  # 使用Asia/Shanghai时区的时间
        feeds=[f['feed'] for f in feeds_data]  # 只传递 feed 部分
    )
    f.write(html)

# 生成聚合的 atom.xml
if feeds_data:
    generate_atom_feed(feeds_data)
