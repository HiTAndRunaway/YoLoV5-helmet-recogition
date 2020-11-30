/*
SQLyog Ultimate v12.08 (64 bit)
MySQL - 5.5.62 : Database - helmet
*********************************************************************
*/

/*!40101 SET NAMES utf8 */;

/*!40101 SET SQL_MODE=''*/;

/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;
CREATE DATABASE /*!32312 IF NOT EXISTS*/`helmet` /*!40100 DEFAULT CHARACTER SET utf8 */;

USE `helmet`;

/*Table structure for table `helmetdata` */

DROP TABLE IF EXISTS `helmetdata`;

CREATE TABLE `helmetdata` (
  `resultId` int(11) NOT NULL AUTO_INCREMENT,
  `helmet` int(50) DEFAULT NULL,
  `nohelmet` int(50) DEFAULT NULL,
  `date` datetime DEFAULT NULL,
  PRIMARY KEY (`resultId`)
) ENGINE=InnoDB AUTO_INCREMENT=10 DEFAULT CHARSET=utf8 ROW_FORMAT=DYNAMIC;

/*Data for the table `helmetdata` */

/*Table structure for table `user` */

DROP TABLE IF EXISTS `user`;

CREATE TABLE `user` (
  `id` int(64) NOT NULL AUTO_INCREMENT COMMENT '主键',
  `username` varchar(100) DEFAULT NULL COMMENT '用户名',
  `password` varchar(100) DEFAULT NULL COMMENT '密码',
  `gender` varchar(10) DEFAULT '' COMMENT '性别',
  `mark` varchar(255) DEFAULT NULL COMMENT '个性签名',
  `avatarUrl` varchar(255) DEFAULT NULL COMMENT '头像地址',
  PRIMARY KEY (`id`) USING BTREE
) ENGINE=InnoDB AUTO_INCREMENT=10 DEFAULT CHARSET=utf8 ROW_FORMAT=DYNAMIC;

/*Data for the table `user` */

insert  into `user`(`id`,`username`,`password`,`gender`,`mark`,`avatarUrl`) values (1,'lyy','111111','女','生活是一种态度','/assets/'),(2,'cky','111111','男','','/assets/'),(3,'zz','111111','男','','/assets/');

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;
