Problem (linear): Implementing the linear module (1 point)  
Deliverable: Implement a Linear class that inherits from torch.nn.Module and performs a linear transformation. 
Your implementation should follow the interface of PyTorch’s built-in nn.Linear module, except for not having a bias argument or parameter. We recommend the following interface:  
def __init__(self, in_features, out_features, device=None, dtype=None) Construct a linear transformation module. This function should accept the following parameters:  
in_features: 
int final dimension of the input  
out_features: int final dimension of the output  
device: torch.device | None = None Device to store the parameters on  dtype: torch.dtype | None = None Data type of the parameters  
def forward(self, x: torch.Tensor) -> torch.Tensor Apply the linear transformation to the input.  
Make sure to:  • subclass nn.Module  • call the superclass constructor  • construct and store your parameter as W (not W ⊤) for memory ordering reasons, putting it in an nn.Parameter  • of course, don’t use nn.Linear or nn.functional.linear  For initializations, use the settings from above along with torch.nn.init.trunc_normal_ to initialize the weights. To test your Linear module, implement the test adapter at [adapters.run_linear]. The adapter should load the given weights into your Linear module. You can use Module.load_state_dict for this purpose. Then, run uv run pytest -k test_linear.

任务要求

目标：实现一个无偏置的线性层 Linear，功能与 nn.Linear 类似，但不支持/不包含 bias。
需继承：torch.nn.Module，并在 __init__ 开头调用 super().__init__()。
接口：
__init__(self, in_features, out_features, device=None, dtype=None)
in_features: 输入最后一维大小（int）
out_features: 输出最后一维大小（int）
device: 参数存放设备（可选）
dtype: 参数数据类型（可选）
forward(self, x: torch.Tensor) -> torch.Tensor
输入形状：(..., in_features)，允许任意前导批维
输出形状：(..., out_features)

参数命名与存储：
仅一个权重参数 W（没有 bias）
存储形状为 (out_features, in_features)（注意是 W，而不是 W^T），放在 nn.Parameter 中
前向计算：
线性变换：y = x @ W.T（或等价的 einsum），正确对齐并保留所有批维
初始化：
使用给定设置并结合 torch.nn.init.trunc_normal_ 初始化权重 W（截断正态分布）
禁止事项：
不得使用 nn.Linear 或 torch.nn.functional.linear
测试与适配器：
在 adapters.run_linear 中实现测试适配器：加载给定权重到你的 Linear（可用 Module.load_state_dict）
运行测试：uv run pytest -k test_linear
验收要点

形状对齐正确：输入 (..., in_features) 输出 (..., out_features)，支持任意批维
W 为 Parameter、形状 (out_features, in_features)，位于期望的 device/dtype
初始化用 trunc_normal_
不依赖内置 nn.Linear/F.linear
通过 test_linear 测试
需要我帮你检查 forward 里的形状处理和 dtype/device 细节，或一并补上 adapters.run_linear 的适配实现思路吗？