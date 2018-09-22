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