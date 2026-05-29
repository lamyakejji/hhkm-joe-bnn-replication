###--------------------------------------------------------------------------###
###---------- Bayesian Neural Networks for Macroeconomic Analysis -----------###
###--------------- Hauzenberger, Huber, Klieber & Marcellino ----------------###
###--------------------------- Replication Code -----------------------------###
###--------------------------------------------------------------------------###
rm(list=ls())
w.dir <- ""

###--------------------------------------------------------------------------###
###-------------------------- Setup & Directories ---------------------------###
###--------------------------------------------------------------------------###
# Create the results directory if it doesn't exist yet
if(!dir.exists("results")) dir.create("results")

library(MASS)
library(RcppArmadillo)
library(Rcpp)
library(coda)
library(stochvol)
library(Matrix)
library(mvtnorm)
library(truncnorm)
library(zoo)
library(dplyr)

source(paste0(w.dir, "functions/aux_funcs_deepNN.R"))
Rcpp::sourceCpp(paste0(w.dir,"functions/get_post_k.cpp"))
Rcpp::sourceCpp(paste0(w.dir,"functions/get_post_grad_k.cpp"))

###--------------------------------------------------------------------------###
###----------------------- Master Configurations ----------------------------###
###--------------------------------------------------------------------------###
mcmc.setup <- list(
  "nsave"   =  10,
  "nburn"   =  10,
  "nthin"   =  1,
  "nuts"    =  FALSE     
)

# Application preliminaries 
target <- "INDPRO_mom"   # Target variable
info   <- "L"            # Information set
stdz   <- TRUE           # Standardize data 

# Model preliminaries (Locked to shallow-NN-flex per your request)
bnn.type <- "shlwNNflex" 
M.lbl    <- M <- 20
cons     <- FALSE
sv       <- TRUE; bsig.SV <- 0.2

NN.layer <- 1
act.flex <- TRUE     

Q <- NN.layer

bnn.setup <- list(
  "sv"          = sv,
  "nsave"       = mcmc.setup$nsave,
  "nburn"       = mcmc.setup$nburn,
  "nthin"       = mcmc.setup$nthin,
  "nuts"        = mcmc.setup$nuts,  
  "main.spec"   = bnn.type,         
  "Q"           = Q,                
  "M"           = M.lbl,            
  "act.flex"    = act.flex,         
  "bsig.SV"     = bsig.SV,          
  "t0.sig"      = 3,                
  "S0.sig"      = 0.3               
)

###--------------------------------------------------------------------------###
###----------------- Out-of-Sample (OOS) Master Loop Setup ------------------###
###--------------------------------------------------------------------------###
oos_dates <- seq(2000, 2000 + 1/12, 1/12)
total_months <- length(oos_dates)

# Storage for final metrics
actual_holdouts <- rep(NA, total_months)
point_forecasts <- rep(NA, total_months)
predictive_densities <- matrix(NA, nrow = mcmc.setup$nsave, ncol = total_months)

cat("Starting Recursive Forecasting: 2000 to 2020\n")
cat("Total iterations:", total_months, "\n")
cat("---------------------------------------------------\n")

start_time_total <- Sys.time()

for (t in 1:total_months) {
  
  hout <- oos_dates[t]
  cat(sprintf("[%d/%d] Estimating hold-out period: %.3f...\n", t, total_months, hout))
  
  ###------------------------------------------------------------------------###
  ###----------------------------- Data setup -------------------------------###
  ###------------------------------------------------------------------------###
  # Sourcing this inside the loop so X, y, Xho, and yho dynamically update 
  # as the holdout period moves forward in time.
  source(paste0(w.dir, "functions/data_designmat.R"))
  
  if(M.lbl == "K") M <- K else M <- as.numeric(M)
  bnn.setup$M <- M
  
  ###---------------------- MCMC preliminaries ------------------------------###
  list2env(bnn.setup, globalenv())
  
  acf_set <- act.fc.set.all[-which(names(act.fc.set.all)=="uni")] 
  if (grepl("relu",main.spec)) acf_set <- acf_set[which(names(acf_set) == "relu")]
  
  ntot <- nburn+nsave*nthin
  save.set <- seq(nthin, nsave*nthin, nthin) + nburn
  save.ind <- 0
  
  XX <- X
  QQ <- Q + 1          
  K  <- ncol(XX)         
  N  <- nrow(XX)         
  R  <- length(acf_set) 
  
  ###------------ Get design matrices, priors, and starting values ----------###
  MM <- c(K, rep(M, Q))
  k_draw <- k.V <- array(0,dim=c(K,M,Q))
  y.hat <- array(0,dim=c(N,M,Q)); yho.hat <- array(0,dim=c(Nho,M,Q))
  X.hat <- array(0,dim=c(N,K,QQ)); Xho.hat <- array(0,dim=c(Nho,K,QQ))
  X.hat[,,1] <- X; Xho.hat[,,1] <- Xho
  
  k_draw[,,1] <- t(matrix(runif(K*M, 0, 1), M, K))
  y.hat[,,1]   <- X.hat[,1:M,2]   <- X.hat[,,1]%*%as.matrix(k_draw[,,1])/K
  yho.hat[,,1] <- Xho.hat[,1:M,2] <- Xho.hat[,,1]%*%as.matrix(k_draw[,,1])/K
  k.V[,,1] <- matrix(1e-10, K, M)
  
  XX.hat<- cbind(X,X.hat[,1:M,QQ])
  KM <- K+M
  
  b_draw  <-  matrix(0,M,1) 
  b.v_draw <- rep(10^2,M)
  acc.k <- array(0, c(1, M, Q))
  post.fc <- matrix(0,  R, 1) 
  
  lambda.beta.mat <- matrix(0.1, M, 1)
  nu.beta.mat <- matrix(0.1,  M, 1)
  tau.beta <- 0.1
  zeta.beta <- 0.1
  
  lam.mat <- nu.mat <- tau.mat <- zeta.mat <- list()
  lam.mat[[1]]  <- matrix(0.1, K, M)
  nu.mat[[1]]   <- matrix(0.1,  K, M)
  tau.mat[[1]]  <- matrix(0.1, M, 1)
  zeta.mat[[1]] <- matrix(0.1, M, 1)
  
  obs_draw <- solve(crossprod(XX.hat) + 1e-5*diag(ncol(XX.hat)))%*%crossprod(XX.hat, y)
  fit_full <- fit_lin <- fit_nn <- XX.hat%*%obs_draw
  fit_nn[] <- 0
  g_draw <-  matrix(0,K,1)
  sigma2.ols <- crossprod(y - XX.hat%*%obs_draw)/(N-KM)
  g.v <- rep(1, K)
  b.v <- rep(1, M)
  g.v.inv <- 1/g.v 
  b.v.inv <- 1/b.v 
  
  g.lam.mat <- matrix(0.1, K, 1)
  g.nu.mat  <- matrix(0.1,  K, 1)
  g.tau     <- 0.1
  g.zeta    <- 0.1
  
  if(sv){
    sv_priors <- specify_priors(
      mu = sv_normal(mean = 0, sd = 10), 
      phi = sv_beta(25, 1.5), 
      sigma2 = sv_gamma(shape = 0.5, rate = 1/(2*bsig.SV)), 
      nu = sv_infinity(),
      rho = sv_constant(0))
    svdraw <- list(mu = 0, phi = 0.99, sigma = 0.01, nu = Inf, rho = 0, beta = NA, latent0 = 0)
  }else{
    t0 <- t0.sig
    S0 <- S0.sig
  }
  
  sig2_draw <- rep(1,N)*as.numeric(0.1)
  ht_draw <- log(sig2_draw)
  acf_draw <- rep(1,M) 
  nuts.eps <- rep(0.0001,Q)
  par_list <- list(list(M_adapt = 1, M_diag = NULL))[rep(1,M)]
  par_list <- list(par_list)[rep(1,Q)]
  
  pred_store <- matrix(NA, nsave, Nho)
  
  ###-------------------- START: MCMC estimation loop -----------------------###
  for (irep in seq_len(ntot)){
    
    # Step 1: Sample gamma and beta
    norm.sig <- 1/sqrt(sig2_draw) 
    y.lin <- y*norm.sig
    x.lin <- cbind(X, X.hat[,1:M,QQ])*norm.sig
    
    gb.V_po <- try(solve(crossprod(x.lin) + diag(c(g.v.inv, b.v.inv))), silent=F) 
    if (is(gb.V_po,"try-error")) gb.V_po <- ginv(crossprod(x.lin) + diag(c(g.v.inv, b.v.inv)))
    gb.m_po <- gb.V_po%*%crossprod(x.lin, y.lin) 
    gb_draw <- try(gb.m_po + t(chol(gb.V_po))%*%rnorm(K+M), silent=F) 
    if (is(gb_draw, "try-error")) gb_draw <- matrix(as.numeric(mvtnorm::rmvnorm(1, gb.m_po, as.matrix(forceSymmetric(gb.V_po)))), K+M,1)
    
    g_draw <- gb_draw[1:K,,drop = F]
    b_draw <- gb_draw[(K+1):(K+M),, drop = F]
    
    fit_lin <- X%*%g_draw 
    y.nolin <- y - fit_lin
    
    # Step 2: Sample prior variances for gamma (HS prior)
    g_hs <- get.hs(g_draw,lambda.hs = g.lam.mat, nu.hs = g.nu.mat, tau.hs = g.tau,zeta.hs=g.zeta)
    g.lam.mat <- g_hs$lambda  
    g.nu.mat  <- g_hs$nu
    g.tau     <- g_hs$tau     
    g.zeta    <- g_hs$zeta
    g.v       <- g_hs$psi     
    g.v[g.v < 1e-15] <- 1e-15 
    g.v.inv <- 1/g.v          
    
    # Step 3: Sample HS prior variances for beta
    hs.beta <- get.hs(b_draw,lambda.hs = lambda.beta.mat, nu.hs = nu.beta.mat, tau.hs = tau.beta,zeta.hs=zeta.beta)
    lambda.beta.mat <- hs.beta$lambda 
    nu.beta.mat <- hs.beta$nu
    tau.beta <- hs.beta$tau           
    zeta.beta <- hs.beta$zeta
    b.v <- hs.beta$psi                
    b.v[b.v > 1]     <- 1             
    b.v[b.v < 1e-10] <- 1e-10         
    b.v.inv <- 1/b.v                  
    v.obs <- c(g.v, b.v)
    v.obs.inv <- c(g.v.inv, b.v.inv)
    
    # Step 4 & 5: Sample kappa and HS priors
    for (nr1 in seq_len(Q)){
      for (nr2 in seq_len(M)){
        theta <- k_draw[1:MM[nr1],nr2,nr1]
        X.hat.nr <- X.hat
        wonr.slct <- setdiff(1:M, nr2)
        X.hat.wonr <- X.hat[,wonr.slct,,drop=F]
        b_draw.nr <- b_draw[nr2,,drop=F]
        b_draw.wonr <- b_draw[wonr.slct,,drop=F]
        k_draw.nr <- k_draw[,nr2,,drop=F]
        k_draw.nr[1:MM[nr1],,nr1] <- theta
        acf_draw.nr <- acf_draw[nr2] 
        
        k_star <- hmc_deep(theta   = theta,
                           f       = get_post_k,        
                           grad_f  = get_post_grad_k,   
                           f_list  = list(k_draw.nr=k_draw.nr,y=y.nolin,X=XX,X.hat.nr=X.hat.nr,X.hat.wonr=X.hat.wonr,k.V=k.V[1:MM[nr1],nr2,nr1],nr1=nr1,nr2=nr2,QQ=QQ, Q=Q,MM=MM,acf_draw=acf_draw.nr,b_draw.nr=b_draw.nr,b_draw.wonr=b_draw.wonr,sig2_draw=sig2_draw,acf_set=acf_set),
                           epsilon = nuts.eps[nr1],
                           L = 20)
        
        accept <- !identical(as.vector(k_star),as.vector(k_draw[,nr2,nr1]))
        if(accept){
          k_draw[1:MM[nr1],nr2,nr1] <- k_star 
          y.hat[,nr2,nr1] <- X.hat[,1:MM[nr1],nr1]%*%k_star    
          yho.hat[,nr2,nr1] <- Xho.hat[,1:MM[nr1],nr1]%*%k_star 
          acc.k[,nr2,nr1] <- acc.k[,nr2,nr1]+1
        }
      } 
      
      for (j in 1:M){
        k_hs.j <- get.hs(bdraw     = k_draw[1:MM[nr1],j,nr1],
                         lambda.hs = lam.mat[[nr1]][,j], 
                         nu.hs     = nu.mat[[nr1]][,j], 
                         tau.hs    = tau.mat[[nr1]][j,1],
                         zeta.hs   = zeta.mat[[nr1]][j,1])
        lam.mat[[nr1]][,j]       <- k_hs.j$lambda 
        nu.mat[[nr1]][, j]       <- k_hs.j$nu
        tau.mat[[nr1]][j,1]      <- k_hs.j$tau    
        zeta.mat[[nr1]][j,1]     <- k_hs.j$zeta
        k.V[1:MM[nr1],j,nr1]     <- k_hs.j$psi    
      } 
      k.V[,,nr1][k.V[,,nr1] > 10] <- 10
      k.V[,,nr1][k.V[,,nr1] < 1e-10] <- 1e-10
    } 
    
    # Step 6: Activation function indicators (shlwNNflex)
    for (nr2 in seq_len(M)) {
      for (rr in seq_len(R)){
        X.hat[ , nr2, QQ] <- acf_set[[rr]][["func"]](y.hat[,nr2,1])
        fit.nr <- as.matrix(X.hat[,1:MM[QQ],QQ])%*%b_draw
        lik <- sum(dnorm(y.nolin, fit.nr, sqrt(sig2_draw), log = TRUE)) 
        prior <- log(1/R) 
        post.fc[rr] <- lik + prior 
      }
      probs <- exp(post.fc -max(post.fc))/sum(exp(post.fc -max(post.fc)))  
      fc.slct <- sample(1:R, 1, prob = as.numeric(probs)) 
      X.hat[,nr2,QQ] <- acf_set[[fc.slct]][["func"]](y.hat[,nr2,1]) 
      Xho.hat[,nr2,QQ] <- acf_set[[fc.slct]][["func"]](yho.hat[,nr2,1]) 
      acf_draw[nr2] <- fc.slct
    }
    
    fit_nn   <- X.hat[,1:M,QQ]%*%b_draw
    fit_nn <- fit_nn - mean(fit_nn)
    
    # Step 7: Error Variances (SV)
    fit_full  <- fit_lin + fit_nn
    eps <-  y - fit_full    
    
    if(sv){
      svdraw <- svsample_fast_cpp(eps, startpara = svdraw, startlatent = ht_draw, priorspec = sv_priors)
      svdraw[c("mu", "phi", "sigma", "nu", "rho")] <- as.list(svdraw$para[, c("mu", "phi", "sigma", "nu", "rho")])
      ht_draw   <- t(svdraw$latent)  
      ht_draw[ht_draw < -10] <- -10
      sig2_draw <- exp(as.numeric(ht_draw)) 
    }else{
      t1 <- t0 + N/2 
      S1 <- S0 + as.numeric(crossprod(eps))/2 
      sig2_cons <- 1/rgamma(1, t1, S1) 
      sig2_draw <- rep(sig2_cons, N) 
      ht_draw <- log(sig2_draw) 
    }
    sig2_draw[sig2_draw > 20*sd(y)] <- 20*sd(y) 
    
    # Step 8: Storage (OOS Predictions Only)
    if(irep %in% save.set){
      save.ind <- save.ind + 1
      pred_m   <- as.numeric(Xho%*%g_draw + Xho.hat[,1:M,QQ]%*%b_draw)  
      if(sv){
        pred_h <- svdraw$para[,"mu"] + svdraw$phi*(ht_draw[N] - svdraw$mu) + rnorm(1, 0, svdraw$sigma)
        pred_V <- exp(pred_h)
      }else{
        pred_V <- sig2_draw[N] 
      }
      pred_draw <- pred_m + rnorm(Nho,0,sqrt(pred_V)) 
      pred_store[save.ind,] <- pred_draw 
    }
  } # End of 3000 MCMC loop
  
  ###------------------------------------------------------------------------###
  ###----------- Extract Final Forecasts for the current month --------------###
  ###------------------------------------------------------------------------###
  # yho should be generated by data_designmat.R for the current hout
  actual_holdouts[t] <- as.numeric(yho) 
  point_forecasts[t] <- mean(pred_store)
  predictive_densities[, t] <- pred_store[, 1]
  
} # End of 20-year expanding window loop

end_time_total <- Sys.time()
cat("\nEstimation Complete. Total Time:", round(as.numeric(difftime(end_time_total, start_time_total, units="mins")), 2), "mins\n")


###--------------------------------------------------------------------------###
###---------------------- Calculate Table 2 Metrics -------------------------###
###--------------------------------------------------------------------------###

# 1. Calculate Root Mean Squared Error (RMSE)
squared_errors <- (actual_holdouts - point_forecasts)^2
overall_rmse <- sqrt(mean(squared_errors, na.rm = TRUE))

# 2. Calculate Log Predictive Likelihood (LPL)
lpl_scores <- rep(NA, total_months)
for (t in 1:total_months) {
  # Empirical Kernel Density Estimation for robust likelihood extraction
  density_est <- density(predictive_densities[, t])
  # Ensure we don't hit log(0) if actual is in extreme tail
  likelihood <- approx(density_est$x, density_est$y, xout = actual_holdouts[t], yleft = 1e-10, yright = 1e-10)$y
  lpl_scores[t] <- log(likelihood)
}
overall_lpl <- mean(lpl_scores, na.rm = TRUE)

# Write Table 2 results to text file
table2_file <- "results/Table2_Metrics.txt"
writeLines(c("Table 2: Forecast Performance (Jan 2000 - Dec 2020)",
             "Target: INDPRO_mom",
             "Model: shlwNNflex",
             "---------------------------------------------------",
             paste("Overall RMSE:", round(overall_rmse, 3)),
             paste("Average LPL: ", round(overall_lpl, 3))), 
           con = table2_file)
cat("Table 2 Metrics saved to:", table2_file, "\n")

###--------------------------------------------------------------------------###
###---------------------- Generate Rolling Plots ----------------------------###
###--------------------------------------------------------------------------###

plot_dates <- as.yearmon(oos_dates)

# Only calculate and plot rolling metrics if we have at least 12 months of data
if (total_months >= 12) {
  
  rolling_rmse <- sqrt(rollapply(squared_errors, width = 12, FUN = mean, align = "right", fill = NA))
  rolling_lpl  <- rollapply(lpl_scores, width = 12, FUN = mean, align = "right", fill = NA)
  
  png("results/Rolling_RMSE_INDPRO.png", width = 800, height = 500, res = 120)
  plot(plot_dates, rolling_rmse, type = "l", col = "darkred", lwd = 2,
       main = "Rolling RMSE: shlwNNflex (Industrial Production)",
       ylab = "RMSE", xlab = "Year")
  grid()
  dev.off()
  
  png("results/Rolling_LPL_INDPRO.png", width = 800, height = 500, res = 120)
  plot(plot_dates, rolling_lpl, type = "l", col = "darkblue", lwd = 2,
       main = "Average LPL: shlwNNflex (Industrial Production)",
       ylab = "Log Predictive Likelihood", xlab = "Year")
  grid()
  dev.off()
  
  cat("Rolling plots saved to results folder.\n")
  
} else {
  
  # For short tests, just use dummy NA vectors to prevent saveRDS from crashing later
  rolling_rmse <- rep(NA, total_months)
  rolling_lpl  <- rep(NA, total_months)
  cat("Notice: Test window too short (< 12 months) for rolling plots. Skipped plotting.\n")
  
}

###--------------------------------------------------------------------------###
###----------------- Proof of Concept: Variable Importance ------------------###
###--------------------------------------------------------------------------###
# Ensure we have the column names from your FRED-MD dataset
var_names <- colnames(X)
if(is.null(var_names)) var_names <- paste0("Var_", 1:ncol(X))

# 1. Extract the linear weights (g_store)
# We take the absolute value because a strong negative effect is just as 
# "important" as a strong positive effect.
abs_weights <- abs(g_store)

# 2. Calculate the posterior mean across all 1000 MCMC draws
posterior_means <- colMeans(abs_weights)

# 3. Create a clean data frame for sorting
importance_df <- data.frame(
  Variable = var_names,
  Importance = posterior_means
)

# 4. Sort to find the Top 15 absolute drivers
top_drivers <- importance_df[order(-importance_df$Importance), ][1:15, ]

# 5. Plot the Results and Save to /results
png("results/Variable_Importance_INDPRO.png", width = 800, height = 600, res = 120)

par(mar = c(5, 8, 4, 2) + 0.1) # Adjust margins for long variable names
barplot(rev(top_drivers$Importance), 
        names.arg = rev(top_drivers$Variable), 
        horiz = TRUE, 
        las = 1, 
        col = "steelblue", 
        border = NA,
        main = "Top 15 Macroeconomic Drivers (Linear Shrinkage)",
        xlab = "Posterior Mean of Absolute Weight")
grid(nx = NULL, ny = NA, col = "gray", lty = "dotted")

dev.off() # This safely closes the file and saves it to the hard drive

cat("Variable Importance plot successfully saved to: results/Variable_Importance_INDPRO.png\n")

###--------------------------------------------------------------------------###
###------------------------- Save Master Raw Data ---------------------------###
###--------------------------------------------------------------------------###
# Save the full arrays in case you need to rebuild the plots later
master_results <- list(
  dates = oos_dates,
  actuals = actual_holdouts,
  point_forecasts = point_forecasts,
  densities = predictive_densities,
  rmse_series = rolling_rmse,
  lpl_series = rolling_lpl
)
saveRDS(master_results, file = "results/OOS_Master_shlwNNflex.rds")
cat("Raw Data saved to: results/OOS_Master_shlwNNflex.rds\n")