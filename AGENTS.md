# AI 开发约束文档

> 本文档是项目开发的硬性约束，AI 和开发者必须严格遵守。

## 文档结构

- **本文件：** 通用规范和部署要求
- **[backend/AGENTS.md](./backend/AGENTS.md)：** 后端开发规范
- **[frontend/AGENTS.md](./frontend/AGENTS.md)：** 前端开发规范（含CSS）

---

## 部署要求

- Python 后端服务的命令必须用 `uv` 包装，禁止使用 `pip`/`pip3`/`python`/`python3` 等命令

---

## 修改要求

- **禁止 AI 自动修改任何 AGENTS.md 文件**

---

## 通用代码规范

### 基本要求

- 每个方法不能超过 **30 行**
- 文件、模块要做到**高内聚、低耦合**、功能逻辑清楚单一
- **新建方法前必须先检查是否已存在类似方法，能复用就复用，禁止无脑新建重复方法**

### 方法复用检查清单

新建任何方法/函数/组件前，必须完成以下检查：

1. **搜索现有代码库** - 使用关键词搜索是否已有类似功能
2. **评估复用可能性** - 优先通过参数化、抽象化改造现有方法
3. **合理性判断** - 确认新方法与现有方法的差异足够大，且会被多处调用

**示例：**

```python
# ❌ 错误 - 重复方法
async def get_active_users():
    return await db.execute(select(User).where(User.status == 'active'))

async def get_verified_users():
    return await db.execute(select(User).where(User.status == 'verified'))

# ✅ 正确 - 参数化复用
async def get_users_by_status(status: str):
    return await db.execute(select(User).where(User.status == status))
```

```typescript
// ❌ 错误 - 重复函数
export const formatUserName = (name: string) => name.trim().toUpperCase()
export const formatProductName = (name: string) => name.trim().toUpperCase()

// ✅ 正确 - 通用函数
export const formatName = (name: string) => name.trim().toUpperCase()
```

---

## 相关文档

- **后端规范：** [backend/AGENTS.md](./backend/AGENTS.md)
- **前端规范：** [frontend/AGENTS.md](./frontend/AGENTS.md)
