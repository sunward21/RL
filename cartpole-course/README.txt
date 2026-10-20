小车不倒 · 读懂强化学习

使用方式
直接用浏览器打开 index.html。若移动或分享，请保留整个 cartpole-course 目录。
断网也可阅读与操作；Google Fonts 无法加载时会使用本机后备字体。

课程内容
1. 让一根杆站稳：命令与用户体验
2. 谁在负责什么：组件分工与群聊动画
3. 四个数字的旅程：逐步数据流与回合边界
4. 学习、保存与重载：模型的生命周期
5. 怎样相信实验结果：对照评估与故障定位
共 15 道场景题、6 处原样源码摘录、42 个术语解释。

内容依据
/home/hxy/RL/cartpole_ppo/cartpole.py、README.md、run.sh、setup.sh、requirements.txt
以及已有 outputs/training.json、random_eval.json、ppo_eval.json 和 cartpole_ppo.gif。
课程于 2026-10-14 制作，展示的是项目记载的 2026-09-24 实验。
没有重新训练模型，也没有改动源项目及其实验输出。

维护方式
修改 modules/ 中的章节后，在本目录执行 bash build.sh 重建 index.html。
styles.css、main.js、build.sh、_footer.html 原样来自 codebase-to-course 技能模板。
course-zh.js 和 course-zh.css 提供中文界面、阅读布局与交互辅助。

检查说明
已通过 HTML 结构、源码逐字一致、资源文件、动画数据、测验配置及 JavaScript 语法检查。
当前会话无可连接的浏览器，未执行浏览器视觉与交互实测。

零基础修订：补充智能体与写代码的 agent 区别、状态与观测、奖励与回报切换示例、策略概率、探索、价值与优势、PPO 裁剪的直觉。先读 1、3、4 章，再读 2、5 章。示意概率和 3/8 步示例并非实际测量数据。
