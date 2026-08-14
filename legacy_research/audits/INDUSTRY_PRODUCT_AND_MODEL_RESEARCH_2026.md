# 智能按摩椅多模态决策：竞品、技术与产品路线调研

> 调研日期：2026-08-12  
> 项目阶段：工业模型规划与重构准备；尚未进入新数据训练  
> 证据口径：优先采用厂商说明书、官方技术页、标准机构与同行评审论文。厂商营销中未公开算法、频率或训练数据的能力，仅记为“厂商声明”，不当作已验证闭环。

## 一、执行结论

**结论：GO，但应把产品定义为“低频多模态状态表征与受约束动作推荐系统”，而不是端到端大模型直接控制机芯。**

当前方向成立，而且与市售主流相比有辨识度：画像、生命体征、舌面诊结果、脑电派生指标与中医体质先验共同形成用户状态表征；系统在有限动作空间中推荐按摩模式、三档力度或 HOLD，并针对缺失模态保持可用。历史 OPLRI 权重能够作为低成本迁移起点，但旧 ECG/EDA 任务、冻结门控及新输入语义之间存在域差异，必须由公司数据上的同预算对照证明其价值。

市场上没有可靠公开证据表明旗舰按摩椅使用“端到端多模态大模型持续直接控制全部按摩动作”。成熟方案普遍分为三层：

1. 入座前或低频状态层：体型扫描、主观疲劳、生理评分、档案偏好，推荐有限程序。
2. 确定性策略与预编程程序层：限制动作目录、切换条件、驻留时间、人工覆盖。
3. 高频机电闭环层：压力、电流、位置、温度与限位反馈，由控制器直接保障力控与安全。

因此，我们真正应追求的不是“模型直接控电机”，而是：**更好的用户状态表示、更可靠的缺模态决策、更快的小样本迁移，以及可记录和改进的个体偏好闭环。**

## 二、同类旗舰产品在做什么

| 产品/厂商 | 可验证的感知与推荐 | 运行中控制 | 对我们的启示 |
|---|---|---|---|
| Panasonic Real Pro EP-MA121/110 | 肩位、位置、揉压传感；官方称揉球压力每秒反馈修正 100 次 | 局部压力与位置高频闭环，用户选择程序/部位/记忆 | “实时”主要指接触力控，不是生理模型连续换模式。[发布](https://panasonic.jp/topics/2024/09/000000936.html) / [技术](https://panasonic.jp/massage/products/EP-MA110/feature/technology.html) |
| Fujiiryoki AS-R2300 | 体型检测；用户点选疲劳部位、程度和目的，5D-AI Navigation 推荐课程 | 厂商称按肌肉负荷调力度、速度和节奏；细节未公开 | 专家动作库仍是核心。官方披露按摩师以 0.1 秒粒度人工标定程序。[产品](https://www.fujiiryoki.co.jp/company/news/news2/n206.html) / [开发](https://www.fujiiryoki.co.jp/product/massagechair/particular/master.html) |
| Family Inada 爱の手/CALABO | 自动找指压点，以揉球负载推断硬度/凝结 | 负载大时深慢、松开后调整节奏 | 接触反馈适合硬件层，不应由上位多模态模型代替。[官方](https://www.family-inada.jp/view/category/ainote) |
| OSIM uDream 系列 | 旧代约 1 分钟 ECG 测 HR/呼吸/HRV；新代以 rPPG 形成 stress 等分数并推荐程序 | 公开证据不足以证明按摩中按生理指标连续改变底层力度 | 派生生理值适合前测/低频推荐，与我们的数据形态接近。[产品](https://us.osim.com/products/udream-ai-well-being-massage-chair) / [手册](https://www.osim.com/Sites/user_manual/udream_en.pdf) |
| 荣泰 RT8900/RT9000 | 官方宣传 HR、SpO2、疲劳指数、酸痛检测、家庭档案和个性化方案 | 三轴机芯压力感知、程序/APP/语音；训练方法未公开 | 国内竞品已把“健康评分+推荐”做成卖点，但算法透明度低。[RT8900](https://www.rotai.com/wap/index.php?ac=article&at=read&did=2565) / [RT9000](https://www.rotai.com/index.php?ac=article&at=read&did=3306) |
| Human Touch Super Novo | 肩位、腿长检测；大量自动程序及手动参数 | 用户调强度、速度、宽度、区域、热疗和气囊 | 自然语言 AI 更多是推荐入口；复杂遥控强化“一键推荐、设置持久”的价值。[手册](https://www.humantouch.com/pub/media/TechnicalSupport/Super-Novo/UserGuide/supernovo_useandcare.pdf) |
| Luraco i9 Max Plus | 身体扫描、用户档案、血压监测 | 专利展示用电机负载电流和位移推断轮廓并控制轨迹 | 即使无压力阵列，电流与位置也可构成低成本硬件反馈。[手册](https://luracochairs.com/wp-content/uploads/2026/02/i9-MAX-PLUS-SERIES-USER-MANUAL-ver2.pdf) / [专利](https://patents.google.com/patent/US10905624) |
| iRest V8 A801-36 | 20 个自动程序 | 机械臂力度/速度/宽度 5 档、气囊 5 档、总体 Gentle/Soothing/Strong | 第一版输出“有限模式+三档总体力度+HOLD”与产品真实交互相符。[手册](https://irest-europe.com/wp-content/uploads/iRest-Dual-Core-v8-A801-36-EN-manual.pdf) |

相邻趋势是 3D 身体建模、双臂轨迹规划、自然语言推荐和用户实时调压。例如 Aescape 将身体扫描与轨迹规划结合，但机电力控仍依赖确定性反馈；2026 年 HMR-1 研究提出分层视觉语言按摩机器人，同时承认缺少统一基准和开放数据。[Aescape](https://time.com/7094771/aescape/) / [HMR-1](https://arxiv.org/abs/2603.08817)

### 市场共通操作流程

`识别用户/加载档案 → 体型扫描 → 主观选择或短时生理前测 → 推荐有限程序 → 用户确认 → 程序执行与局部硬件闭环 → 用户随时升/降/换/停 → 保存偏好与结束反馈`

公开资料很少支持“按摩过程中依据 EEG、HR 或 SpO2 连续自动切换程序”。这意味着我们的低频实时推荐可以形成创新，但必须保守上线，并与硬件闭环解耦。

## 三、别人如何开发和训练

按摩椅厂商几乎不公开训练集和模型细节。能够确认的成熟开发范式是：按摩师/专家设计动作库并细粒度标定；工程团队标定体型、接触负载与执行器；推荐层使用规则或未披露分类算法；量产后累积档案和偏好。行业不是先训练一个大模型，再让它自由生成电机动作。

对本项目，合理的学习对象应是“有限候选动作的接受度与结果”，而不是复制当前正在执行的动作：

1. 每个决策点保存因果历史窗口、当前动作、候选集与输入质量。
2. 分开记录 `requested_action` 与椅子确认的 `applied_action`。
3. 记录用户接受/撤销、手动升降/换模式/停止、动作曝光时长。
4. 在固定延迟窗采集舒适变化、生理派生值变化和不适事件。
5. 在安全动作子集内做均衡或随机交叉试验，保存展示概率；否则只能学到旧规则展示过的动作。

仅用“被选择的模式”训练，得到的是行为模仿，不等于最佳动作。首版可以预测 `accepted_next_action`，另设 action-conditioned outcome 预测接受、舒适变化与不适风险，再枚举有限动作排序。真人在线 RL 不适合第一版；积累充分动作覆盖、propensity 与结果日志后，才考虑离线上下文 bandit 和 doubly robust 评估。[Doubly Robust OPE](https://www.microsoft.com/en-us/research/publication/doubly-robust-policy-evaluation-and-learning-2/)

## 四、近期模型技术趋势及取舍

| 趋势 | 研究进展 | 对当前项目的取舍 |
|---|---|---|
| 缺失多模态表征 | ADAPT 用 anchor alignment + masked Transformer；MUSE 用患者-模态图和对比学习 | 数据规模未知，MVP 先用硬 mask、整模态 dropout 和轻量融合；完整配对/未标注会话足够后再上。[ADAPT](https://arxiv.org/abs/2407.03836) / [MUSE](https://proceedings.iclr.cc/paper_files/paper/2024/hash/f49d76cf84df83a611883c621c96d2d9-Abstract-Conference.html) |
| 轻量时序基础模型 | Tiny Time Mixers、TSMixer 强调小模型、CPU 与多分辨率 | 它们主要做预测，不能硬套派生标量；未来真实压力/原始 EEG 到位后，与 TCN/旧 1D-ResNet 对照。[TTM](https://research.ibm.com/publications/tiny-time-mixers-ttms-fast-pre-trained-models-for-enhanced-zerofew-shot-forecasting-of-multivariate-time-series--1) / [TSMixer](https://research.google/pubs/tsmixer-an-all-mlp-architecture-for-time-series-forecasting/) |
| 生理信号基础模型与个性化 | PhysioPFM 研究低秩个体适配；NormWear 等探索多生理信号预训练 | 当前只有派生值且 OPLRI 体量小，分阶段解冻比 LoRA 更简单；未来原始波形和个体校准数据足够时再评估。[PhysioPFM](https://proceedings.mlr.press/v267/wu25ah.html) |
| 校准与拒绝 | 神经网络常过度自信，温度缩放是简单强基线 | mode/intensity 分别校准；低置信、OOD、过期或缺失过多时 HOLD。[Temperature Scaling](https://arxiv.org/abs/1706.04599) |
| 持续学习/RL | 适合长期漂移与大规模策略日志，但验证和安全成本高 | V1 周期性离线再训练、版本化发布、全局回退；不上在线持续学习和真人在线 RL。 |

这里最重要的原则是：**硬 availability mask 先于可学习 Gate；quality 与 missing 是两件事；按 person/session 整体隔离训练与验证。** 所有 scaler 仅拟合训练人群，最终测试保留完全新用户，并报告 macro-F1、各类召回、用户间方差、缺模态降幅、校准和端侧延迟，而不是只报 accuracy。

## 五、与当前方案逐项对比

### 已经做对的部分

- 复用 OPLRI 预训练，而非从零堆大模型，符合数据少、原型快、算力有限的工业现实。
- 模态独立编码、统一表征、Gate A/B 与 Late Reinjection 可形成清晰的先验调制链路。
- 舌面诊/脑电/生命体征是派生结果值，使用轻量标量 Adapter，而不是伪装成原始波形。
- 压力模态保留接口但首版屏蔽，为后续真实压力时序留出局部扩展点。
- 输出限定为少数 `program_id + intensity{轻柔,舒适,强劲} + HOLD`，避免直接预测底层执行器。
- TCM 作为辅助先验而不是控制器，产品定位更稳健。

### 尚不能证明或必须修正的部分

- OPLRI 的 Gate 学自 ECG/EDA 语义；新输入主要是派生静态/低频结果，shape 可加载不代表语义可迁移。
- 旧实验中的 dynamic detach/no_grad 可能使编码器/门控实际未更新，不能把旧指标直接解释成工业预训练能力。
- Gate B 若改为质量门控，其能力必须重新学习；不能声称直接继承。
- Late Reinjection 可能形成 TCM shortcut，需做无 TCM、缺 diagnostic 等消融。
- 当前还没有公司数据、最终模式目录、硬件实际状态回读与延迟结果，因此只能称“推荐原型规划”，不能称实时闭环已经成立。
- 旧论文数据和指标不能作为新公司数据上的选型依据；它们只提供候选架构和初始化来源。

### 最低限度实验矩阵

同一数据切分、训练预算和预训练来源下，跑：

- M0：mask-aware 融合，无 Gate，保留 Late Reinjection。
- M1：M0 + Gate A。
- M2：M1 + Gate B。
- M2-from-scratch：结构同 M2，不加载 OPLRI，用来回答预训练到底是否省数据/加快收敛。
- 缺模态测试：缺 1 字段、缺 2 字段、缺整个 diagnostic/TCM/neuro 模态及低质量输入。

这四组不是为了“最后留三个模型”，而是以最小代价回答三个产品问题：门控有没有用、双门控是否值得复杂度、预训练是否真的有价值。最终只留一个部署候选。

## 六、推荐的第一版产品与训练流程

### 产品运行链路

1. 加载用户画像与禁忌；椅端负责体型/肩位扫描。
2. 采集可用的生命体征、舌面诊结果、脑电派生值和 TCM 体质；显示缺失/质量状态。
3. 多模态模型形成 `PatientRepresentation`，输出有限模式、三档力度或 HOLD 的候选与置信度。
4. 确定性策略裁剪非法组合，检查输入新鲜度、最短驻留、连续确认、相邻力度与人工覆盖。
5. 用户确认后下发；记录椅子实际执行状态而非只记命令。
6. 运行中按固定低频窗口或事件触发重新建议；多数周期应 HOLD，降力/停止优先。
7. 会话结束采集短反馈、舒适变化和是否愿意再次使用，并写入下一版离线训练集。

### 训练流程

- Stage 0：鉴定 checkpoint 的 key/shape、schema、scaler、冻结状态与父 commit，生成严格加载报告。
- Stage 1：冻结语义匹配的旧层，只训练新模态 Adapter、mask/fusion 与新决策 head。
- Stage 2：以新层约 0.1 倍学习率解冻 Gate 和 Late Reinjection；必要时再逐层解冻后部。
- 训练期执行字段级和整模态 dropout，缺失率尽量贴近真实设备。
- 采用 person/session 隔离的 GroupKFold/留人验证；独立校准集做 temperature scaling。
- 最终候选必须同时通过新用户性能、缺模态鲁棒性、校准、推理延迟和不适/拒绝指标。

### 上线路线

`P0 固定保守程序验证安全与采集 → P1 安全子集受控交叉 → P2 成对偏好 → P3 Shadow 只建议不执行 → P4 人在环小流量 → P5 受约束自动`

模型发布物必须绑定 checkpoint hash、代码 commit、schema/scaler、字段顺序、动作 catalog、安全策略、固件兼容范围和验证数据 manifest。回滚应原子恢复整包，而不是只替换 `.pth`。

## 七、安全、合规和系统边界

模型不能进入 MCU 安全回路。推荐职责为：模型建议约 0.5–1 Hz；策略层约 10–50 Hz；MCU 根据实际硬件带宽约 100–1000 Hz 完成位置、速度、电流/力、限位和看门狗。这里的频率只是工程起点，不是标准阈值，最终应由具体椅型、固件与人体 pilot 确认。

急停、离席、堵转、过流、过温、限位冲突、通信超时和夹困保护必须独立于模型与高级操作系统。人工减力/停止永远优先；重新入座不得自动恢复旧高强度；模型失效时回退固定安全程序。

可参考 [IEC 60335-2-32:2024](https://webstore.iec.ch/en/publication/85019)、[ISO 12100](https://www.iso.org/standard/51528.html)、[ISO 13849-1:2023](https://www.iso.org/standard/73481.html)、[GB/T 26182-2022](https://openstd.samr.gov.cn/bzgk/std/newGbInfo?hcno=87FB789F5261E12DBC3484ADC04B83DA) 与 [ISO/IEC 23894:2023](https://www.iso.org/standard/77304.html)。标准公开页只证明适用范围，完整条款须由公司合规与认证团队购标、解析并落地。

首版应定位为舒适、放松和个性偏好推荐，不宣称诊断、筛查、治疗或临床级监测。FDA 的 [General Wellness 指南](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/general-wellness-policy-low-risk-devices) 可作边界参考，但不能替代中国及目标市场的正式法规判断。

## 八、优先级路线图

### 现在：不写核心逻辑前必须完成

1. 完成服务器 OPLRI checkpoint 的安全鉴定和逐层迁移表。
2. 与硬件团队确认“模式 ID + 总体三档力度”的命令、ACK、实际状态回读、延迟、故障和人工覆盖接口。
3. 确定第一版 3–5 个安全模式，但不要在硬件目录未定前虚构类别。
4. 冻结表示层输入契约：字段语义、单位、availability、quality、时间戳和 schema 版本。
5. 明确模型、策略、MCU 三层边界以及日志事件时间线。

### 数据到位后的 MVP

1. 先完成 M0/M1/M2/M2-from-scratch 同预算验证。
2. 以新用户、缺模态、校准和延迟选择唯一候选，不以旧论文指标选型。
3. 先上线 Shadow，再人在环；无硬件状态回读时只做推荐演示。
4. 建立 action-conditioned outcome 与不适风险记录，为后续个性化积累数据。

### 数据和产品稳定后

- 压力原始时序到位后，比较旧 1D-ResNet/TCN；再决定是否引入 TSMixer/TTM。
- 完整配对和大量不完整会话足够时，评估 ADAPT/MUSE 一类对齐或对比预训练。
- 每用户校准样本足够且 backbone 变大后，再评估 adapter/LoRA 个性化。
- 有可靠 propensity、动作覆盖和延迟结果后，先做 offline contextual bandit；不直接上真人在线 RL。

## 九、最终判断

我们的方案不是“落后于最火产品”，也不需要为了显得先进而推翻旧架构。它抓住了市场尚未充分解决的一层：将多个低频人体状态结果、TCM 先验和缺失信息统一成可迁移表征，再对有限动作进行个性化推荐。

但现阶段竞争力仍是**有依据的架构假设**，不是已验证产品能力。能否成立，取决于五项证据：

1. OPLRI 预训练相对 from-scratch 是否更快、更省数据或更准。
2. Gate A/B 和 Late Reinjection 是否在公司数据上产生稳定净收益。
3. 新用户与缺模态条件下是否优于简单规则/Simple baseline。
4. 是否获得椅端真实执行状态、人工覆盖和延迟效果反馈。
5. 是否以 Shadow→人控→受约束自动的流程守住安全和回滚边界。

只要按这条路线推进，当前方案值得继续，而且比盲目追逐“大模型直控按摩椅”更接近可量产的工业原型。
