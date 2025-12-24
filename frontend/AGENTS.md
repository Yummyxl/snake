# 前端开发规范

> 返回 [总规范](../AGENTS.md)

## 三层架构

```
View层 → Store层 → API层 → 后端
               ↑
        Utils ←┘
```

| 层级 | 职责 | 可调用 | 禁止 |
|------|------|--------|------|
| **View** | UI渲染、用户交互 | Store、Utils | API（除简单查询）、复杂业务逻辑（>5行）、操作其他组件状态 |
| **Store** | 全局状态、业务逻辑 | API、Utils | 直接fetch、操作DOM、UI状态（弹窗显示等） |
| **API** | 封装接口、错误转换 | Utils | 业务逻辑、修改Store、缓存数据 |
| **Utils** | 纯函数工具 | 无 | API、Store、全局状态、副作用 |
| **Hooks** | 复用组件逻辑 | Store、Utils | 直接调API |

---

## View 层

**核心：** 只做渲染和调用Store，业务逻辑都在Store

```tsx
// ✅ 正确
function OrderList() {
  const { orders, loading, fetchOrders } = useOrderStore()
  useEffect(() => { fetchOrders() }, [])
  return <div>{loading && <Spinner />}{orders.map(...)}</div>
}

// ❌ 错误
function OrderList() {
  const [orders, setOrders] = useState([])
  useEffect(() => {
    api.getOrders().then(res => {  // 直接调API
      const filtered = res.filter(o => o.status === 'active' && ...)  // 业务逻辑
      setOrders(filtered)
    })
  }, [])
}
```

**检查清单：** 新建组件前检查是否可通过props参数化复用

---

## Store 层

**核心：** 管理状态 + 业务逻辑，通过API层调接口

```typescript
// ✅ 正确
const useOrderStore = create((set, get) => ({
  orders: [],
  loading: false,

  fetchOrders: async () => {
    set({ loading: true })
    const orders = await orderApi.getList()
    set({ orders, loading: false })
  },

  cancelOrder: async (orderId: string) => {
    const order = get().orders.find(o => o.id === orderId)
    if (order.status !== 'pending') throw new Error('只能取消待处理订单')
    await orderApi.cancel(orderId)
    set(state => ({ orders: state.orders.map(o => o.id === orderId ? { ...o, status: 'cancelled' } : o) }))
  }
}))

// ❌ 错误
const useOrderStore = create((set) => ({
  fetchOrders: async () => {
    const res = await fetch('/api/orders')  // 直接fetch
    set({ orders: await res.json() })
  }
}))
```

**检查清单：** 新建Store方法前检查是否可参数化复用

---

## API 层

**核心：** 只封装接口，不做业务逻辑

```typescript
// ✅ 正确
const orderApi = {
  getList: async (params?: OrderQueryParams): Promise<Order[]> => {
    const res = await request.get('/orders', { params })
    return res.data
  },
  cancel: async (id: string): Promise<void> => {
    await request.post(`/orders/${id}/cancel`)
  }
}

// ❌ 错误
const orderApi = {
  getActiveOrders: async () => {
    const res = await request.get('/orders')
    return res.data.filter(o => o.status === 'active')  // 业务过滤
  }
}
```

**检查清单：** 新建API方法前检查是否可参数复用（统一CRUD）

---

## Utils 层

**核心：** 纯函数，无副作用

```typescript
// ✅ 正确
export const formatPrice = (price: number): string => `¥${(price / 100).toFixed(2)}`
export const validatePhone = (phone: string): boolean => /^1[3-9]\d{9}$/.test(phone)

// ❌ 错误
export const formatPrice = (price: number): string => {
  const currency = localStorage.getItem('currency')  // 访问外部状态
  return `${currency}${price}`
}
```

**检查清单：** 新建工具函数前检查utils目录和第三方库（lodash、date-fns等）

---

## Hooks 层

**核心：** 封装可复用逻辑，组合多个Store

```typescript
// ✅ 正确
function useOrderWithUser(orderId: string) {
  const order = useOrderStore(state => state.orders.find(o => o.id === orderId))
  const user = useUserStore(state => state.users[order?.userId])
  const displayName = useMemo(() =>
    order && user ? `${user.name}的订单 #${order.id.slice(0, 8)}` : '',
    [order, user]
  )
  return { order, user, displayName }
}
```

**检查清单：** 新建Hook前检查React官方hooks、第三方库（ahooks、react-use等）

---

## 文件组织

```
src/
├── views/              # 页面组件
├── components/         # 全局组件
├── stores/             # 状态管理
├── api/                # 接口层
├── hooks/              # 公共hooks
├── utils/              # 工具函数
├── styles/             # 样式文件
│   ├── variables.css
│   ├── global.css
│   └── mixins.css
└── types/              # 类型定义
```

---

# CSS 规范

## 模块化策略

| 方案 | 使用场景 |
|------|----------|
| **CSS Modules** | 组件私有样式、复杂布局 |
| **Tailwind CSS** | 快速样式、间距、颜色、简单布局 |

---

## CSS Modules

**限制：** 每组件对应一个`.module.css`、禁止全局选择器、禁止业务逻辑命名、**重复样式必须复用**

```tsx
import styles from './OrderCard.module.css'
function OrderCard({ order }) {
  return <div className={styles.card}>...</div>
}
```

```css
/* OrderCard.module.css */
.card {
  padding: var(--spacing-md);
  border-radius: var(--radius-md);
  background: var(--color-bg-card);
}

/* ❌ 错误 - 业务命名 */
.vipUser { color: gold; }

/* ✅ 正确 - 视觉命名 */
.textGold { color: gold; }
```

---

## Tailwind CSS

**限制：** 优先使用Tailwind、禁止配置业务样式、**重复超过3次必须抽取**

```tsx
// ✅ 正确 - 简单样式
<button className="px-4 py-2 bg-blue-500 text-white rounded">提交</button>

// ❌ 错误 - 重复3次
<div className="flex items-center gap-4 p-4 border rounded">...</div>
<div className="flex items-center gap-4 p-4 border rounded">...</div>
<div className="flex items-center gap-4 p-4 border rounded">...</div>

// ✅ 正确 - 抽取为CSS类
<div className={styles.card}>...</div>
```

```css
/* styles.module.css */
.card { @apply flex items-center gap-4 p-4 border rounded; }
```

---

## CSS 变量

**限制：** 颜色/字体/间距必须用变量、禁止硬编码、变量命名 `--category-name-variant`、**新建前检查复用**

```css
/* styles/variables.css */
:root {
  --color-primary: #3b82f6;
  --color-bg-card: #ffffff;
  --color-text-primary: #111827;

  --spacing-sm: 8px;
  --spacing-md: 16px;
  --spacing-lg: 24px;

  --font-size-base: 14px;
  --font-size-lg: 16px;

  --radius-md: 8px;
  --shadow-md: 0 4px 6px rgba(0, 0, 0, 0.1);
}

/* ❌ 错误 */
.button { background: #3b82f6; padding: 16px; }

/* ✅ 正确 */
.button { background: var(--color-primary); padding: var(--spacing-md); }
```

---

## 样式优先级

1. **Tailwind** → 简单、通用样式
2. **CSS Modules** → 组件特定、复杂布局
3. **全局样式** → 仅重置、全局字体、基础标签
4. **内联样式** → 仅动态计算值

---

## 响应式设计

**断点：** `--breakpoint-sm: 640px` (手机) / `md: 768px` (平板) / `lg: 1024px` (笔记本) / `xl: 1280px` (桌面)

**策略：** Mobile First + Tailwind响应式前缀

```tsx
<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
  {/* 移动端1列，平板2列，桌面3列 */}
</div>
```

---

## 性能优化

| 禁止 | 原因 | 推荐 |
|------|------|------|
| `@import` | 阻塞渲染 | ES6 import |
| 选择器>3层 | 性能差 | 扁平化 `.navLink` |
| 通配符 `*` | 性能差 | 具体选择器 |
| `transition: width` | 触发重排 | `transition: transform` |

```css
/* ❌ 错误 */
.header .nav .menu .item .link { }
.box { transition: width 0.3s; }

/* ✅ 正确 */
.navLink { }
.box { transition: transform 0.3s; }
.box:hover { transform: scaleX(1.2); }
```

---

## 暗色模式

**实现：** CSS变量 + `data-theme`

```css
/* styles/variables.css */
:root {
  --color-bg-page: #ffffff;
  --color-text-primary: #111827;
}

[data-theme="dark"] {
  --color-bg-page: #1f2937;
  --color-text-primary: #f9fafb;
}
```

```tsx
function App() {
  const [theme, setTheme] = useState('light')
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
  }, [theme])
  return <div>...</div>
}
```
