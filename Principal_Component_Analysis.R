library(ggplot2)
library(dplyr)
library(MASS)
library(caret)

  f_read <- read.csv("/Users/PCA/mnist_train.csv")
  
  m_read <- data.matrix(f_read) ; dim(m_read)
  
  # Actual digit represented by xn
  dig_A <- 2  # negative
  # Actual digit represented by xp
  dig_B <- 6  # positive
  
  X0 <- subset(m_read, m_read[,1] == dig_B | m_read[,1] == dig_A); dim(X0)
  
  #Sample Row where my digit is A & B
  digitA <- 3097 
  digitB <- 6097 
  
  #remove the class lables in the first column
  X <- X0[,-1] ;  dim(X); min(X); max(X)
  
  # function to display the row as an image 
  picimage <- function(testmatrix)    image(t(apply(testmatrix, 2, rev)),col=gray(12:1/12))
  
  #test my image at rowA 3097(2)
  testmatrix <- matrix(X[digitA,],28,28,byrow = T);picimage(testmatrix)
  
  #test my image at rowB 6097(6)
  testmatrix <- matrix(X[digitB,],28,28,byrow = T);picimage(testmatrix)
  
  # mu (mean vector)
  MeanX <- colMeans(X); min(MeanX);max(MeanX);dim(MeanX)

  # Z Matrix:
  MeanVector <- matrix(MeanX,dim(X)[1],784,byrow = TRUE)
  
  Z <- X-MeanVector; min(Z); max(Z); dim(Z)
  
  # Covariance: 
  C <- cov(Z); dim(C) 
  
  # check how Covariance image looks like
  image((C))
  
  #alternate way of calculating covariance
  #C_check <- (t(Z) %*% Z)/((dim(X)[1])-1)
  
  # find eigen value and eigen vecotr
  V <- eigen(C, symmetric=TRUE);dim(V$vectors)
  VM <- V$vectors;  
  
  VM <- t(VM)
  
  # Principal component Matrix 
  P <- Z %*% t(VM); dim(P)
 
  # Full Reconstructed R & ZP Matrix
  R0 <- (P %*%VM) ; dim(R0)
  XR0 <- R0 + MeanVector
  
  #test my image at fully Reconstructed row @ 3098 & 9360
  testmatrix <- matrix(XR0[digitB,],28,28,byrow = T);picimage(testmatrix)
  testmatrix <- matrix(XR0[digitA,],28,28,byrow = T);picimage(testmatrix)
  
  # Reconstructed R & ZP Matrix with only 2 PCAS and two eigen vectors
  R <- (P[,1:2] %*%VM[1:2,]) ; dim(R)
  XR <- R + MeanVector
  
  #test my image at Reconstructed with only 2 PCA @ 3098 & 9360
  testmatrix <- matrix(XR[digitB,],28,28,byrow = T);picimage(testmatrix)
  testmatrix <- matrix(XR[digitA,],28,28,byrow = T);picimage(testmatrix)
  
  # xrecp (reconstructed xp)
  xreconstructed_B <-  XR[digitB,]
  
  # xrecn (reconstructed xn)
  xreconstructed_A <-  XR[digitA,]
  
#####################################################################################################

#****************************************************************************************************
# Plot the 2 PCAS for the digit A & B

  # take only the first two PCAs and their class lables and build a Data Frame
  newdf   <- data.frame(X0[,1],P[,1], P[,2])
  colnames(newdf) <- c("digit", "P1", "P2")
  
  df_B    <- dplyr::filter(newdf,digit==dig_B) %>% select(P1,P2); count(df_B)
  df_A    <- dplyr::filter(newdf,digit==dig_A) %>% select(P1,P2); count(df_A)
  
  # plot my first two PCAS for the digits
  plot(newdf$P1,newdf$P2, col=newdf$digit)
  ggplot(newdf, aes(x=newdf$P1, y=newdf$P2, color=newdf$digit)) +  geom_point() + theme(panel.background = element_rect(fill = 'gray'))
  

 #############################################################################
  # Histogram distribution for the two digits
  
  # Histogram range. pc1 direction
  Min_P1Vector=min(newdf$P1); Max_P1Vector=max(newdf$P1);Max_P1Vector;Min_P1Vector
  # Histogram range. pc2 direction
  Max_P2Vector=max(newdf$P2); Min_P2Vector=min(newdf$P2);Max_P2Vector;Min_P2Vector

  # Bin Size
  P1BinsCount <- 25
  P2BinsCount <- 25


  myBHistarray <- array(0, c(P1BinsCount, P2BinsCount))
  myAHistarray <- array(0, c(P1BinsCount, P2BinsCount))

for (i in 1:(dim(newdf)[1]))
  
{
  if(newdf$digit[i] == dig_B)
  {
    a= (1+((P1BinsCount -1)*((newdf$P1[i]-Min_P1Vector)/( Max_P1Vector-Min_P1Vector))));a
    b= (1+((P2BinsCount -1)*((newdf$P2[i]-Min_P2Vector)/( Max_P2Vector-Min_P2Vector))));b
    
    myBHistarray[a,b]<- myBHistarray[a,b]+1 ; myBHistarray[a,b]
  }
  
  if(newdf$digit[i] == dig_A)
  {
    a= round(1+((P1BinsCount -1)*((newdf$P1[i]-Min_P1Vector)/( Max_P1Vector-Min_P1Vector))));a
    b= round(1+((P2BinsCount -1)*((newdf$P2[i]-Min_P2Vector)/( Max_P2Vector-Min_P2Vector))));b
    
    myAHistarray[a,b]<- myAHistarray[a,b]+1 ; myAHistarray[a,b]
  }
  
}
 dim(myAHistarray); sum(myAHistarray)+ sum(myBHistarray)
 
 #check how my histigram looks
 image(myAHistarray); image(myAHistarray)
 image(myBHistarray); image(myBHistarray)
#*************************************************************************************************
#*************************************************************************************************

# check Histogram Probability

  row_B <- newdf[digitB,]
  
  r= round(1+((P1BinsCount -1)*((row_B$P1-Min_P1Vector)/( Max_P1Vector-Min_P1Vector))));r
  c= round(1+((P2BinsCount -1)*((row_B$P2-Min_P2Vector)/( Max_P2Vector-Min_P2Vector))));c
  
  # Result of classifying xp using histograms
  myBProb <- myBHistarray[r,c]/(myBHistarray[r,c]+myAHistarray[r,c]);myBProb
 
  row_A <- newdf[digitA,]
 
  
  r= round(1+((P1BinsCount -1)*((row_A$P1-Min_P1Vector)/( Max_P1Vector-Min_P1Vector))));r
  c= round(1+((P2BinsCount -1)*((row_A$P2-Min_P2Vector)/( Max_P2Vector-Min_P2Vector))));c
  
  # Result of classifying xn using histograms
  myAProb <- myAHistarray[r,c]/(myBHistarray[r,c]+myAHistarray[r,c]);myAProb


##################################################################################################################
  #Naive Bayes Probability
  
  # B= positive and A=negative
  # Np (class +1 number of samples) and Nn (class -1 number of samples)
  N_PB    <- dim(df_B)[1]; N_PB ;      N_PA <- dim(df_A)[1]; N_PA
  
  # mup (class +1 mean vector)
  mean_PB <- colMeans(df_B); mean_PB
  
  # mun (class -1 mean vector)
  mean_PA <- colMeans(df_A); mean_PA
  
  min_P2_B <- min(df_B$P2);   min_P2_B;       max_P1_B  <- max(df_B$P1) ; max_P1_B     
  min_P2_A <- min(df_A$P2);   min_P2_A;       max_P1_A <- max(df_A$P1); max_P1_A
  

  #Covariance Matrix
  # cp (class +1 covariance matrix) and cn (class -1 covariance matrix)
  Cov_A <- cov(cbind(df_A$P1, df_A$P2)); Cov_A
  Cov_B <- cov(cbind(df_B$P1, df_B$P2)); Cov_B

  UA <- mean_PA; UA
  UB <- mean_PB; UB
  
  detA <- det(Cov_A); detA
  detB <- det(Cov_B); detB
  
  
  xb <- cbind(newdf$P1[digitB],newdf$P2[digitB]);xb
  
  FB <- N_PB*(1/(2*3.14*sqrt(detB)))*exp((-1/2)*((xb-UB)%*%solve(Cov_B)%*%t(xb-UB)))
  MA <- N_PA*(1/(2*3.14*sqrt(detA)))*exp((-1/2)*((xb-UA)%*%solve(Cov_A)%*%t(xb-UA)))
  # Result of classifying xp using Bayesian
  ProbXpBayes <- FB/(FB+MA)
  
  xa <- cbind(newdf$P1[digitA],newdf$P2[digitA]);xa
  
  FB <- N_PB*(1/(2*3.14*sqrt(detB)))*exp((-1/2)*((xa-UB)%*%solve(Cov_B)%*%t(xa-UB)))
  MA <- N_PA*(1/(2*3.14*sqrt(detA)))*exp((-1/2)*((xa-UA)%*%solve(Cov_A)%*%t(xa-UA)))
  # Result of classifying xn using Bayesian
  ProbXnBayes <- MA/(FB+MA)
  
############################################################################################
  
  
#############################################################################################
# We are doing the accurancy now, we have to built a new column with actual value and precicted value
#############################################################################################
  
  myXProbV=newdf
  # i have to it for histogram and bayes propability...
  for(i in 1:(N_PA+N_PB)){
    
    r= (1+((P1BinsCount -1)*((myXProbV$P1[i]-Min_P1Vector)/( Max_P1Vector-Min_P1Vector))))
    c= (1+((P2BinsCount -1)*((myXProbV$P2[i]-Min_P2Vector)/( Max_P2Vector-Min_P2Vector))))
    
    myBHistarray[a,b]<- myBHistarray[a,b]+1 ; myBHistarray[a,b]
    
    myXProbV$prob[i] <- myBHistarray[r,c]/(myBHistarray[r,c]+myAHistarray[r,c]);
    
  }
  
  myXProbV$predicted <- ifelse(myXProbV$prob >= 0.5, dig_B, dig_A) 

  
  confusionMatrix(myXProbV$predicted, myXProbV$digit)
  
############################################################################################
  # Accurancy with Naive Bayes
############################################################################################
  
  myXProbVBayes=newdf
 
  for(i in 1:(N_PA+N_PB)){

    
    xb <- cbind(myXProbVBayes$P1[i],myXProbVBayes$P2[i])
    xb<-as.matrix(xb,byrow=TRUE)
    print(i)    
    print(xb)
    print(UB)
    
    FB <- N_PB*(1/(2*3.14*sqrt(detB)))*exp((-1/2)*((xb-UB)%*%solve(Cov_B)%*%t(xb-UB)))
    
    MA <- N_PA*(1/(2*3.14*sqrt(detA)))*exp((-1/2)*((xb-UA)%*%solve(Cov_A)%*%t(xb-UA)))
   
     # Result of classifying xp using Bayesian
    
    myXProbVBayes$ProbXpBayes[i] <- FB/(FB+MA)

    
  }
  
  myXProbVBayes$predicted <- ifelse(myXProbVBayes$ProbXpBayes >= 0.5, dig_B, dig_A) 
  
  confusionMatrix(myXProbVBayes$predicted, myXProbVBayes$digit)
  
  
