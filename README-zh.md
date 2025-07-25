# RSS-GPT

[![](https://img.shields.io/github/last-commit/yinan-c/RSS-GPT/main?label=feeds%20refreshed)](https://yinan-c.github.io/RSS-GPT/)
[![](https://img.shields.io/github/license/yinan-c/RSS-GPT)](https://github.com/yinan-c/RSS-GPT/blob/master/LICENSE)

如果想要一个网页端 GUI 来更好地管理 feeds，请关注我的最新项目：[RSSBrew](https://github.com/yinan-c/RSSBrew)，一个开源自托管的 RSS-GPT 替代方案，在过滤，自定义 prompt 等方面更加强大。

## 这是什么？

[中文介绍](https://yinan-c.github.io/rss-gpt.html) | [中文教程](https://yinan-c.github.io/rss-gpt-manual-zh.html) | [README](README.md)

使用 GitHub workflow 自动运行一个简单的 Python 脚本，调用 OpenAI API 为 RSS 订阅源生成摘要，然后将新生成的 RSS 订阅源推送到 GitHub Pages。配置简单快速，无需服务器。

### 功能及示例

- 使用 ChatGPT 来总结 RSS 订阅源, 生成关键词，摘要附在原文前面，支持自定义摘要长度，自定义语言。
- 聚合多个 RSS 订阅源，去除重复文章，用单一地址订阅。
- 为 RSS 订阅源添加基于标题，内容，URL 的关键词过滤器。
- 在 GitHub 仓库和 GitHub Pages 上自托管 RSS 订阅源。

![](https://i.imgur.com/7darABv.jpg)

## 快速部署

- Fork 这个仓库中
- 添加仓库 Secrets
    - U_NAME: 你的 GitHub 用户名
    - U_EMAIL: 你的 GitHub 邮箱
    - WORK_TOKEN: 你的 GitHub 个人访问令牌, 需要有 `repo` 和 `workflow` 权限。在 [GitHub 设置](https://github.com/settings/tokens/new)获取
    - OPENAI_API_KEY(可不填, 只有在使用 AI 摘要功能时需要): 你的 OpenAI API 密钥, 在 [OPENAI 网站](https://platform.openai.com/account/api-keys)获取
- 在仓库设置中启用 GitHub Pages， 选择 deploy from branch，设置目录为 `/docs`.
- 在 `config.ini` 中配置你的RSS订阅源

也可以参考更详细的[中文教程](https://yinan-c.github.io/rss-gpt-manual-zh.html)。

## 脚本的更新

- 由于 OpenAI 在 2023-11-06 发布了新版本的 `openai` 包，[新版本包含了更强大的模型](https://openai.com/blog/new-models-and-developer-products-announced-at-devday)，调用 API 的方式也发生了变化。因此，旧版本的脚本将无法使用最新版本的 `openai` 包，需要更新。否则，你可以在 `requirements.txt` 中设置 `openai==0.27.8` 来使用旧版本。
- 查看 [CHANGELOG-zh.md](CHANGELOG-zh.md) 获取其他最新的更新日志。

### 欢迎贡献!

- 欢迎提交 issue 和 pull request。

## 分享几条本项目处理后的 RSS 订阅源

我自己用此脚本总结的一些 RSS订阅源托管在本项目的[`docs/`子目录](https://github.com/yinan-c/RSS-GPT/tree/main/docs)中以及我的 [GitHub Pages](https://yinan-c.github.io/RSS-GPT/)上找到。欢迎在任何 RSS 阅读器中订阅。

如果有任何问题或有关于 RSS feeds 的建议，欢迎邮件联系我。

如果你觉得本项目有帮助，欢迎 star。也可以考虑捐赠以支持我继续维护本项目以及 cover OpenAI API 的支出。感谢支持。

<a href="https://www.buymeacoffee.com/yinan" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" style="height: 60px !important;width: 217px !important;" ></a>

- https://freshrss.chunkj.dpdns.org/i/?a=rss&user=chunkj&token=chunkj&hours=24 -> https://kzy77.github.io/RSS-GPT/首页 | FreshRSS.xml
- https://7techlife.blogspot.com/atom.xml -> https://kzy77.github.io/RSS-GPT/Seven科技生活.xml
