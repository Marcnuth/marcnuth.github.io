#How to install deploy tomcat using HTTPS 443 port

##Environments

- CentOS Linux release 7.1.1503 (Core)
- Apache Tomcat Version 8.0.33
- java version "1.8.0_91"

##Deploy steps
### Download & install tomcat & jdk
### Generate a keystore file
Using keytool to generate keystore file:

```

keytool -genkey -keystore to_generate_filename.keystore -alias anyname_you_like -keyalg RSA

```

After the keystore is generated, you will get:

- a keystore file
- your keystore alias
- your keystore password(This is inputed by you when generating the file)

### Configure tomcat
- Open bin/startup.sh, and insert following lines into the front lines

```shell

export TOMCAT_HOME=/path/to/tomcat/
export CATALINA_BASE=/path/to/tomcat/
export CATALINA_HOME=/path/to/tomcat/
export JAVA_HOME=/path/to/jdk/
export PATH=$JAVA_HOME/bin:$PATH

```

- Open conf/server.xml, change the lines

```xml

<Connector port="8080" protocol="HTTP/1.1"
               connectionTimeout="20000"
               redirectPort="8443" />

<!--
<Connector port="8443" protocol="org.apache.coyote.http11.Http11NioProtocol"
               maxThreads="150" SSLEnabled="true" scheme="https" secure="true"
               clientAuth="false" sslProtocol="TLS"/>
-->

```

to:

```xml


<Connector port="80" protocol="HTTP/1.1"
               connectionTimeout="20000"
               redirectPort="8443" />



<Connector port="443" protocol="org.apache.coyote.http11.Http11NioProtocol"
               maxThreads="150" SSLEnabled="true" scheme="https" secure="true"
               clientAuth="false" sslProtocol="TLS"
               keystoreFile="/path/to/your/keystore/xxx.keystore"
               keystorePass="your_keystore_password"/>


```

### Enjoy
After above things are done, just run bin/startup.sh and enjoy.

