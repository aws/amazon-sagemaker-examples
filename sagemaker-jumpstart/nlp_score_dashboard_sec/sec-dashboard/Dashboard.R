if (!require('DT')) install.packages('DT'); library(DT)

df = read.csv('stock_sec_scores.csv', header=TRUE)
df2 = df[,1:(length(df)-1)]

for (j in 2:dim(df2)[2]) {
  df2[,j] = round(df2[,j],4)
}

dtable <- datatable(df2, filter='top', width = '100%', height = '100%')
saveWidget(dtable, "SEC_Dashboard.html")