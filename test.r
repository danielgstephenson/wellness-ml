m = c()

for(n in 1:10) {
  z = c()
  print(n)
  for(i in 1:100000) {
    x = rnorm(n)
    xbar = mean(x)
    popError = mean((x-0)^2)
    sampleError = mean((x-xbar)^2)
    z[i] = mean(popError - sampleError)
  }
  m[n] = mean(z)
}

plot(x=2:10,y=m[2:10])

x = rnorm(2)

plot(dnorm,xlim=c(-2,2))
abline(v=x,lty='dashed')
abline(v=mean(x),col='blue')
abline(v=0,col='red')

mean(abs(x-mean(x)))
mean(abs(x-0))

