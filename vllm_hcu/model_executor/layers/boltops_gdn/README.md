# 外置 GDN kernel 的兼容导入入口

kernel 实现、辅助函数、gfx936 配置和许可证已经移至独立目录：

`/jsyadmin/guanyu/qwen3_8-next-flash-PD/boltops-gdn-kernel`

本目录只保留旧 Python 导入路径的转发，不包含 kernel 实现或配置文件。
新的直接调用入口为 `from boltops_gdn_kernel import chunk_gated_delta_rule_fwd_h`。
将独立目录加入 `PYTHONPATH` 即可使用，不需要编译或重新安装 HCU。
用法详见独立目录下的 `README.md`。
