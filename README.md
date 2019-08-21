## Lookahead Optimizer for Keras

The [original repository](https://github.com/bojone/keras_lookahead/) is implemented based on Keras > 2.0.8<div>
This repository is for Keras =< 2.0.8
which didn't have following function:
````python
_check_trainable_weights_consistency
````
[commit](https://github.com/keras-team/keras/commit/cab77c8f23bf81eaa06aeeeb28a4da3b716f7bd7)

Keras implement of [Lookahead Optimizer: k steps forward, 1 step back](https://arxiv.org/abs/1907.08610).

Usage:
```
model.compile(optimizer=Adam(1e-3), loss='mse') # Any optimizer
lookahead = Lookahead(k=5, alpha=0.5) # Initialize Lookahead
lookahead.inject(model) # add into model
```

## Lookahead优化器的Keras实现

论文[《Lookahead Optimizer: k steps forward, 1 step back》](https://arxiv.org/abs/1907.08610)的Keras实现。

[原始專案](https://github.com/bojone/keras_lookahead/) 是給Keras >2.0.8的<div>
此專案是給 Keras =< 2.0.8
這些舊版本缺少了像是下列函式:
````python
_check_trainable_weights_consistency
````
[commit](https://github.com/keras-team/keras/commit/cab77c8f23bf81eaa06aeeeb28a4da3b716f7bd7)

用法：
```
model.compile(optimizer=Adam(1e-3), loss='mse') # 用你想用的优化器
lookahead = Lookahead(k=5, alpha=0.5) # 初始化Lookahead
lookahead.inject(model) # 插入到模型中
```

中文介绍：[点击进入](https://mp.weixin.qq.com/s/3J-28xd0pyToSy8zzKs1RA)
