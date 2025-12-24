# 后端开发规范

> 返回 [总规范](../AGENTS.md)

## 三层架构

```
API层 → Service层 → Data层 → Database
                  → External层
```

| 层级 | 职责 | 可调用 | 禁止 |
|------|------|--------|------|
| **API** | 接收请求、参数校验、返回响应 | Service | Data、其他API、业务逻辑、直接数据库操作 |
| **Service** | 业务编排、组合Data、协调External | Data、External | 其他Service、直接SQL/ORM、HTTP异常 |
| **Data** | 单表CRUD | Database | 跨表JOIN、其他Data、业务逻辑、External |
| **External** | 封装第三方API、超时/重试/异常 | HTTP Client | Data、业务逻辑 |

---

## API 层

**核心：** 只做参数校验 → 调用Service → 返回结果，其他都禁止

```python
# ✅ 正确
@router.post("/orders")
async def create_order(req: CreateOrderRequest):
    return await order_service.create_order(req.user_id, req.product_ids)

# ❌ 错误
@router.post("/orders")
async def create_order(req: CreateOrderRequest):
    user = await user_data.get_by_id(req.user_id)  # 直接调data
    if not user.is_vip:  # 业务逻辑
        raise HTTPException(400, "非VIP用户")  # HTTP异常在API层
```

**检查清单：** 新建路由前检查是否可通过路径参数/查询参数复用

---

## Service 层

**核心：** 组合Data操作实现业务逻辑，抛 `BusinessError` 不抛 `HTTPException`

```python
# ✅ 正确
async def create_order(user_id: int, product_ids: list[int]):
    user = await user_data.get_by_id(user_id)
    if not user:
        raise BusinessError("用户不存在")
    products = await product_data.get_by_ids(product_ids)
    return await order_data.create(user_id, sum(p.price for p in products))

# ❌ 错误
async def create_order(user_id: int, product_ids: list[int]):
    user = await db.execute(select(User).where(...))  # 直接ORM
    raise HTTPException(404, "用户不存在")  # HTTP异常
```

**检查清单：** 新建业务方法前检查是否可参数化复用

---

## Service 层 - 外部调用5大规则

| 规则 | 要求 | 示例 |
|------|------|------|
| **1. 先本地后外部** | 先提交本地事务，再调外部服务 | `await db.commit()` → `await external.call()` |
| **2. 超时和重试** | 必须配置 | `timeout=5000, retry_times=3` |
| **3. 记录失败状态** | 失败时记录，支持补偿 | `await data.update_status(id, "failed", error=str(e))` |
| **4. 支持幂等** | 传递幂等键 | `idempotency_key=f"order_{order.id}"` |
| **5. 重要性排序** | 失败影响小的放前面 | 先通知 → 后支付 |

**完整示例：**

```python
async def create_paid_order(user_id: int, amount: int):
    # 1. 本地事务
    order = await order_data.create(user_id, amount, status="pending")
    await db.commit()

    # 2. 外部调用
    try:
        await payment_external.charge(
            order_id=order.id,
            amount=amount,
            idempotency_key=f"order_{order.id}",
            timeout=5000,
            retry_times=3
        )
        await order_data.update_status(order.id, "paid")
    except ExternalError as e:
        await order_data.update_status(order.id, "failed", error=str(e))
        raise
```

---

## Data 层

**核心：** 一个类只操作一张表，只提供基础CRUD

```python
# ✅ 正确
class OrderData:
    async def get_by_id(self, order_id: int) -> Order | None:
        return await db.get(Order, order_id)

    async def create(self, user_id: int, amount: int, status: str) -> Order:
        order = Order(user_id=user_id, amount=amount, status=status)
        db.add(order)
        return order

# ❌ 错误
class OrderData:
    async def get_order_with_user(self, order_id: int):
        return await db.execute(
            select(Order, User).join(User).where(Order.id == order_id)  # 跨表JOIN
        )
```

**检查清单：** 新建CRUD前检查是否可通过可选参数复用

---

## External 层

**核心：** 封装第三方调用，统一异常为 `ExternalError`，必须配置超时

```python
class PaymentExternal:
    async def charge(self, order_id: int, amount: int, idempotency_key: str) -> PaymentResult:
        try:
            resp = await self.client.post(
                "/charge",
                json={"order_id": order_id, "amount": amount, "key": idempotency_key},
                timeout=5.0
            )
            return PaymentResult(**resp.json())
        except httpx.TimeoutException:
            raise ExternalError("支付服务超时")
        except httpx.HTTPError as e:
            raise ExternalError(f"支付服务异常: {e}")
```

**检查清单：** 新建外部调用前检查是否可参数化复用
