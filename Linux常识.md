# 一些linux常识

## 几个配置文件

### 全局

* /etc/profile
  * 此文件为系统的每个用户设置环境信息，全局(公有)配置，不管是哪个用户，登录时都会读取该文件
  * 当用户第一次登录时，该文件被执行，并从/etc/profile.d目录的配置文件中收集shell的设置
  * /etc/profile中设定的变量(全局)的可以作用于任何用户，而~/.bashrc等中设定的变量(局部)只能继承/etc/profile中的变量，他们是“父子”关系
* /etc/bashrc
  * 为每一个运行bash shell的用户执行此文件，当bash shell被打开时，该文件被读取
  * Ubuntu下没有此文件，与之对应的是/etc/bash.bashrc，它也是全局的；bash执行时，不管是何种方式，都会读取此文件

### 局部

* ~/.profile 
  * 若bash是以**login方式**执行时，读取`~/.bash_profile`
  * 若它不存在，则读取`~/.bash_login`
  * 若前两者不存在，读取`~/.profile`
  * 另外，<u>**图形模式**登录时，此文件将被读取，即使存在`~/.bash_profile` 和 `~/.bash_login`</u>

* ~/.bash_profile
  * 每个用户都可使用该文件输入专用于自己使用的shell信息，当用户登录时，该文件仅仅执行一次
  * 默认情况下，它设置一些环境变量，执行用户的`.bashrc`文件，是**交互式、login方式**进入bash运行的，通常二者设置大致相同，所以通常前者会调用后者
* ~/.bash_login
  * bash是以**login方式**执行时，读取`~/.bash_profile`
  * 若它不存在，则读取`~/.bash_login`
  * 若前两者都不存在，则读取`~/.profile`
* ~/.bash_profile 
  * <u>Ubuntu默认没有此文件</u>，可新建
  * 只有bash是以**login形式**执行时，才会读取此文件
  * 通常该配置文件还会配置成去读取`~/.bashrc`
* ~/.bashrc
  * 该文件包含用于你的bash shell的bash信息，当登录时以及每次打开新的shell时，该文件被读取
  * 当bash是以**交互式、non-login**形式执行时，读取此文件
* ~/.bash_logout
  * 当每次退出系统(退出bash shell)时，执行该文件
  * **注销时，且是login形式**，此文件才会读取。<u>也就是说，在文本模式注销时，此文件会被读取，图形模式注销时，此文件不会被读取</u>

### 下面是在本机的例子

1. 图形模式登录时，顺序读取：`/etc/profile`和`~/.profile`

2. 图形模式登录后，打开终端时，顺序读取：`/etc/bash.bashrc`和`~/.bashrc`

3. 文本模式登录时，顺序读取：`/etc/bash.bashrc`，`/etc/profile`和`~/.bash_profile`

4. 从其它用户su到该用户，则分两种情况：
   1. 如果带-l参数（或-参数，--login参数），如：su -l username，则bash是login的，它将顺序读取以下配置文件：`/etc/bash.bashrc`，`/etc/profile`和`~ /.bash_profile`。
   2. 如果没有带-l参数，则bash是non-login的，它将顺序读取：`/etc/bash.bashrc`和`~/.bashrc`

5. 注销时，或退出su登录的用户，如果是longin方式，那么bash会读取：`~/.bash_logout`

6. 执行自定义的shell文件时，若使用“bash -l a.sh”的方式，则bash会读取行：`/etc/profile`和`~/.bash_profile`，若使用其它方式，如：bash a.sh， ./a.sh，sh a.sh（这个不属于bash shell），则不会读取上面的任何文件。

7. 上面的例子凡是读取到`~/.bash_profile`的，若该文件不存在，则读取`~/.bash_login`，若前两者不存在，读取`~/.profile`。

## bash美化配置

> https://www.cnblogs.com/heqiuyu/articles/5624694.html

对于终端的美化,可以通过对PS1变量进行赋值来进行.奶牛查阅了一些资料,DIY了如下的效果:

![bash 美化 bash PS PS1](http://img2.tuicool.com/3Qn6Ff.jpg!web)

分享下奶牛的修改方法:

```sh
vim .bashrc #添加下行
export PS1="Time:\[\033[1;35m\]\T     \[\033[0m\]User:\[\033[1;33m\]\u     \[\033[0m\]Dir:\[\033[1;32m\]\w\[\033[0m\]\n\$"
# 退出vim
source .bashrc
```

解释下具体含义:

颜色配置:

> \[\033[ 1 ; 31 m\]

* 底线 ：ANSI 色彩控制语法。\033 声明了转义序列的开始，然后是 [ 开始定义颜色。
* 第一组数字 ：亮度 (普通0, 高亮度1, 闪烁2)。
* 第二组数字 ：顏色代码。
* 颜色: 30=black 31=red 32=green 33=yellow 34=blue 35=magenta 36=cyan 37=white

> \[\033[0m\]

* 关闭 ANSI 色彩控制，通常置于尾端。

显示内容配置:

* \a     ASCII响铃字符 (07)
* \d     “周 月 日”格式的日期
* \D{format}   参数format被传递给strftime(3)来构造自定格式的时间并插入提示符中；该参数为空时根据本地化设置自动生成格式。
* \e     ASCII转义字符（ESC) (033)
* \h     主机名在第一个点号前的内容
* \H     完全主机名
* \j     shell当前管理的任务数
* \l     shell终端设备的基本名称
* \n     新行
* \r     回车
* \s     shell的名称，$0的基本名称
* \t     当前时间（24小时） HH:MM:SS
* \T     当前时间（12小时） HH:MM:SS
* \@     当前时间（12小时） am/pm
* \A     当前时间（24小时） HH:MM
* \u     当前用户名称
* \v     bash版本(如"2.00")
* \V     bash版本+补丁号(如"2.00.0")
* \w     当前工作目录
* \W     当前工作目录的基本名称
* \!     该命令的历史数（在历史文件中的位置）
* \#     该命令的命令数（当前shell中执行的序列位置）
* \$     根用户为"#"，其它用户为"$"
* \nnn   8进制数
* \\     反斜杠
* \[     表示跟在后面的是非打印字符，可用于shell的颜色控制
* \]     表示非打印字符结束

## [emacs] error: (error "Fontset `tty' does not exist")

```lisp
(cond ((display-graphic-p)
            (set-fontset-font ...)
                       ...                    )
           (t 0))
```

![1538187832584](../assets/1538187832584.png)

## dpkg

```sh
//查询deb包的详细信息，没有指定包则显示全部已安装包
dpkg –l  
dpkg -l |grep vim
//查看已经安装的指定软件包的详细信息
dpkg -s vim
//列出一个包安装的所有文件清单
dpkg -L vim
//查看系统中的某个文件属于那个软件包
dpkg -S vimrc
//所有deb文件的安装
dpkg -i
//所有deb文件的卸载
dpkg -r
//彻底的卸载，包括软件的配置文件
dpkg -P
//查询deb包文件中所包含的文件
dpkg -c
//查看系统中安装包的的详细清单，同时执行 -c
dpkg -L
```

