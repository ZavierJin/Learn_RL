# Tips

> **UserWarning: Your graphics drivers do not support OpenGL 2.0. You may experience rendering issues or crashes.**

Need an older version of pyglet, `pip install pyglet==1.5.11`.

https://github.com/openai/gym/issues/2101

> **RuntimeError: Trying to backward through the graph a second time, but the buffers have already been freed. Specify retain_graph=True when calling backward the first time**

The commom mistake that can happen is that you perform some computation just before or out of the loop, and so even though you create new graphs in the loop, they share a common part out of the loop. You need to use `.detach()` to detach them from the graph.

https://blog.csdn.net/axept/article/details/114370257
