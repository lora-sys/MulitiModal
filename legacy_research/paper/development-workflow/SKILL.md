# Development Workflow

> 按需加载：当需要启动新功能开发时加载此 skill。
> 加载方式：`/skills:development-workflow`

---

## 整体流程

```
git checkout main && git pull origin main
    ↓
git checkout -b feature/xxx
    ↓
specs/hackthon/specs/feature-xxx/
    ├── prd.md     → 用户痛点、核心逻辑、用户故事
    ├── plan.md    → 技术选型、架构设计、数据流向 + GitHub Issue
    ├── task.md    → 原子级 Task List
    ├── check.md   → 验收标准
    ├── analyze.md → 变更分析、Breaking Changes、技术难点
    └── implement.md → 伪代码、函数签名、接口契约
    ↓
实现 → e2e test → lint / build / type-check → PR → merge
    ↓
更新 CHANGELOG.md + 版本号
```

---

## 阶段一：PRD → PLAN

**产出文件**：`hackthon/specs/feature-xxx/prd.md` + `hackthon/specs/feature-xxx/plan.md`

`prd.md` 包含：
- 用户痛点（User Pain Points）
- 核心业务逻辑（Core Business Logic）
- 用户故事（User Stories）
- Out of Scope（明确不做什么）

`plan.md` 包含：
- 技术选型
- 架构设计
- 数据流向
- 在 GitHub/GitLab 创建对应的 Issue

---

## 阶段二：TASK → CHECK

**产出文件**：`hackthon/specs/feature-xxx/task.md` + `hackthon/specs/feature-xxx/check.md`

`task.md` 示例：
```markdown
- [ ] 1. 定义数据类型
- [ ] 2. 实现后端 API
- [ ] 3. 编写前端 UI
- [ ] 4. 编写测试
```

`check.md` 包含：
- 验收标准（Acceptance Criteria）
- 静态检查清单

---

## 阶段三：ANALYZE → IMPLEMENT

**产出文件**：`hackthon/specs/feature-xxx/analyze.md` + `hackthon/specs/feature-xxx/implement.md`

`analyze.md` 包含：
- 哪些现有文件会被修改
- 是否存在 Breaking Changes
- 技术难点及规避方案

`implement.md` 包含：
- 伪代码、关键函数签名
- 接口契约（API Contracts）
- **AI 以该文件为直接依据生成代码**

---

## 阶段四：测试与质量

```bash
# 端到端测试
npx playwright test

# 本地质量门禁
npm run lint       # 代码风格
npm run build      # 编译检查
npm run type-check # 类型检查
```

---

## 阶段五：合并与版本

1. 提交 Pull Request，附带 `hackthon/specs/` 下全部变更记录
2. Code Review 通过后合并到 `main`
3. 更新版本号
4. 在 `CHANGELOG.md` 记录本次 Feature / Fix
