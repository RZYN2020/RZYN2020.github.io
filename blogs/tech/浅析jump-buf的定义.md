---
categories: System
tags:
- C
- Knowledge
date: '2021-11-14T09:23:52'
draft: false
title: 浅析jmp_buf的定义
toc: true
---

<!--more-->
`int setjmp(jmp_buf env)`

`void longjmp(jmp_buf env, int val)`


setjmp 和 longjmp 是`setjmp.h`定义的相互协作的一组跳转函数。 调用 setjmp 时可以将当前的环境保存在一个`jmp_buf`类型的变量中，之后调用 longjmp 后会跳转到 setjmp 执行后的下一条语句执行，就好像刚刚从 setjmp返回一样。
> 函数行为描述见man，源码见[glibc](https://www.gnu.org/software/libc/)。

其中,`jmp_buf`的定义如下:
```c
typedef long int __jmp_buf[8];

/* Calling environment, plus possibly a saved signal mask.  */
struct __jmp_buf_tag
  {
    /* NOTE: The machine-dependent definitions of `__sigsetjmp'
       assume that a `jmp_buf' begins with a `__jmp_buf' and that
       `__mask_was_saved' follows it.  Do not move these members
       or add others before it.  */
    __jmp_buf __jmpbuf;		/* Calling environment.  */
    int __mask_was_saved;	/* Saved the signal mask?  */
    __sigset_t __saved_mask;	/* Saved signal mask.  */
  };


typedef struct __jmp_buf_tag jmp_buf[1];
```
本来预想jmp_buf应该是简单的一个存储寄存器信息的数组，却发现其定义较为复杂。在阅读其定义的时候，又牵扯出了许多不熟悉的c知识点。试解析定义如下：
其中`typedef struct __jmp_buf_tag jmp_buf[1]`定义了一个名为`jmp_buf`的变量类型,它实际上是一个大小为1的`struct __jmp_buf_tag`数组。而结构体`struct __jmp_buf_tag`包含三个成员，后两个与信号机制有关，不做讨论。第一个成员为`__jmp_buf`类型，用来保存寄存器信息。而`__jmp_buf`类型实际上是一个大小为8的`long int`数组。
那么为什么要把实际上存储信息的结构体`__jmp_buf_tag`包含在一个数组里面呢？也许是因为将数组当作参数传递时总是传递数组的地址，而将结构体当作参数传递时却总是将整个结构体的值赋值一遍传给被调用函数。我们的`jmp_buf`作为一个在函数调用间保存信息的实体应该满足数组的特征，因此将其定义为数组更合适一些。当然，如果不这样做，每次被调用函数需要结构体`__jmp_buf_tag`时传入它的指针也是可行的，只是略显麻烦罢了。

> hint:
> 结构体定义了一种变量类型，作为一个整体复制和赋值。在行为上更加类似于int而非int[];
> 变量名是与值绑定的符号，而指针是与一个地址值绑定的符号。

