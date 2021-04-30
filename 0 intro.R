

rm(list=ls())
gc()

library(data.table)
library(mlr3verse)
library(tidyverse)


getwd()
#data <- fread("datasets_2/calibrado_201905.csv")
data <- fread("datasets_2/paquete_premium_201906_202005.csv")

dim(data)
names(data)
View(data)
summary(data)

min(data[, foto_mes])
max(data[, foto_mes])
length(data[, foto_mes])

