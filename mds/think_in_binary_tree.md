# 另一个角度看翻转二叉树

## 起源
很多工程师的面试或笔试中，都可能接触过这样一道题：实现翻转二叉树。

## 满二叉树的翻转
### 数据结构
用数组进行存储，其中，index=0存储元素个数，后面一次为对应节点。
因此，有：
- 二叉树元素个数: array[0]
- 二叉树层数: log2(array[0] + 1) 向上取整
- 二叉树根节点: array[1]
- index=i的节点的可计算信息:
  - 层数：log2(i + 1) 向上取整, 记该层数为x
  - 当前节点的父节点index: i为偶数时: i/2， i为奇数时， (i-1)/2

### 翻转满二叉树代码
Python:

```
# coding: utf-8

def invert(tree):
	pass
	
def __name__ == '__main__':
	tree = [8,18,9,0,3,7,81,4]
	

```
