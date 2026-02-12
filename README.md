Gender Switch: 4,000 Trials of a Murder Case
性别开关：一个故意杀人案审判的4,000次对照实验

同一个案件。同一套事实。同一份量刑情节。
唯一变量：被告性别。

我们向大语言模型提交了两个版本的上海“护士注射胰岛素故意杀人案”：

A 版：被告为女性护士，被害人为男性医生

B 版：被告为男性护士，被害人为女性医生

其他事实——作案动机、预谋过程、手段残忍程度、事后伪造信息、一审死刑判决——完全一致。

重复运行 4,000 次。

🔍 发现
女性被告：死刑率 50%（死刑立即执行 + 死缓）

男性被告：死刑率 86%

死刑立即执行率差距：36 个百分点

同一个法定情节（婚恋矛盾激化、被害人存在暧昧关系），
流向女性时：“情绪崩溃，可不立即执行”
流向男性时：“虽有纠纷背景，不足以从轻”

这不是随机误差。
这是真实判决语料喂养出的性别基准位移。

❓ 这不是在“审判”AI
本项目不旨在“揭露”某个模型有偏见。
模型只是回音壁。它忠实地复现了它学习过的数万份真实刑事判决书中的性别模式。

我们真正映照出的，是中国司法实践中——至少在公开可获取的判例语料中——故意杀人案量刑存在的系统性性别差异。

镜子没有错。
错的是镜子映照出的东西。

📂 仓库包含
两份完全对称的实验刺激材料（仅交换性别代词与亲属称谓）

4,000 次运行的完整输出日志（JSONL 格式）

数据分析脚本（Python, R）

可视化图表（死刑率/死缓率/无期率 分性别条形图）

🧪 复现
bash
git clone https://github.com/yourname/gender-switch-trial
cd gender-switch-trial
pip install -r requirements.txt
python run_experiment.py --trials 1000
📄 引用
若您使用本数据集或发现，请引用：

text
@misc{gender_switch_2026,
  title={Gender Switch: 4,000 Trials of a Murder Case},
  author={Anonymous},
  year={2026},
  howpublished={\url{https://github.com/yourname/gender-switch-trial}}
}
这不是一个关于AI的故事。
这是一个关于我们自己的故事。AI只是把它写了出来。
