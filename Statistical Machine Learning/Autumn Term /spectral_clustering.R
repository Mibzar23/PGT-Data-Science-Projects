# A METHODOLOGICAL INVESTIGATION OF SPECTRAL CLUSTERING IN MACHINE LEARNING

# ----- K-Means in Linear Separable Dataset -----
# The purpose of this section is to do a statistical breakdown of a linear clustering method before developing the non-linear approach.
# Download "clusteringToyData.csv" and uploade it as data-frame.
setwd("C:\\Users\\sebas\\OneDrive - The University of Nottingham\\University of Nottingham - MSc Data Science\\Statistical Machine Learning\\Projects\\sml_project1")
X <- read.csv("clusteringToyData.csv",header=FALSE)

# Visualize the points.
plot(1, 
     type="n", 
     xlab="x_1", 
     ylab="x_2", 
     xlim=c(1,7.8), 
     ylim=c(-0.5,8),
     main="Eleven Toy Dataset"
)
# Since there are few data points, lets assign them numbers to clearly identify them.
text(X[,1], 
     X[,2], 
     pch=as.character(1:11),
     cex=0.9
)
# Add a circle around each number.
points(X[,1], 
       X[,2], 
       pch=1,
       cex=2.7
)

# Define hyperparameters.
n <- 11 # data size.
K <- 3 # number of clusters.

# Initialization.
inds <- sample(1:n, size=K, replace=FALSE) # select three random data points to be centroids.
m <- X[inds,] # create matrix with the initial centroids.
# Plot the selected centroids.
points(m, 
       pch=11, 
       col=c(2,3,4)
)

# Cluster data points based on their distance to the initial centroid with Euclidean distance.
# Define Euclidean distance formula to work with the two matrices.
dis <- function(x1, x2){
  sqrt(sum((x1 - x2)^2)) 
}
# Assigns each point in 'X' to the closest point in 'm'.
group <- matrix(NA,n,1)
for (i in 1:n) {
  group[i] <- which.min(c(dis(X[i,],m[1,]), dis(X[i,],m[2,]), dis(X[i,],m[3,])))
} 

# Colour points according to the assigned cluster.
text(X[,1], 
     X[,2], 
     pch=as.character(1:11), 
     col=group+1
)
points(X[,1], 
       X[,2], 
       pch=1,
       cex=2.7,
       col=group+1
)

# Update mean for each cluster and plot it into the current graph.
for (j in 1:K){
 m[j,] <- colMeans(X[group==j,])
}
points(m,
       pch=4,
       col=c(2,3,4)
)

# Iteration loop to update means until convergance.
conv <- FALSE # convergance identifier.
iter <- 0 # iteration parameter.

while(!conv) {
  iter <- iter + 1
  
  last_m <- m

  for (i in 1:n) {
   group[i] <- which.min(c(dis(X[i,],m[1,]),dis(X[i,],m[2,]),dis(X[i,],m[3,])))
  }

  plot(1, type="n", xlab="x_1", ylab="x_2", xlim = c(1,7.8), ylim=c(-0.5,8), main="Eleven Toy Dataset with K-means (3 Clusters)")
  text(X[,1], X[,2], pch=as.character(1:11))
  points(m, pch=4, col=c(2, 3, 4))
  text(X[,1], X[,2], pch=as.character(1:11), col=group+1)
  points(X[,1], X[,2], pch=1, cex=2.7, col=group+1)

  for (j in 1:K){
   m[j,] <- colMeans(X[group==j,])
  }

  points(m,pch=4, col=c(2,3,4))
  
  conv <- all(m == last_m)
}
# Print the number of iterations required.
cat("Iterations = ", iter)

# ----- Spectral Clustering in Linear Separable Dataset  -----
# In this I use spectral clustering to identify the same 3 clusters in the same Toy dataset used for K-Means.

# Plot a clean graph again.
plot(1, 
     type="n", 
     xlab="x_1", 
     ylab="x_2", 
     xlim=c(1,7.8), 
     ylim=c(-0.5,8),
     main="Eleven Toy Dataset"
)
text(X[,1], 
     X[,2], 
     pch=as.character(1:11),
     cex=0.9
)
points(X[,1], 
       X[,2], 
       pch=1,
       cex=2.7
)

# Spectral clustering technique is based on graph theory for data partitioning.
# Taking a binary approach to create edges, the fist thing is to calculate the adjacency matrix.
adjacency_matrix <- function(X, nn){ # create a formula to calulcate the adjacency matrix.
     n <- nrow(X) # get the number of rows of any matrix.
     dist_matrix <- matrix(data=NA, nrow=n, ncol=n) # square matrix to store the distance of each point to all others.
     # Calculate and store the distance between each point.
     for (i in 1:n){
          for (j in 1:n){
               dist_matrix[i, j] <- dis(X[i,], X[j,])
          }
     }
     adjacency_matrix <- matrix(data=0, nrow=n, ncol=n) # create a square matrix filled with '0s'.
     # Update adjacency matrix column-wise so any '0' is replaced with '1' if any point is part of the 'nn' nearest neighbours of the point in the correponding column.
     for (i in 1:n){
       n_col <- dist_matrix[,i]
       dists.sort <- sort(n_col,  index.return=TRUE)
       inds <- dists.sort$ix[2:(nn+1)]
       adjacency_matrix[i, inds] <- 1  
     }
     adjacency_matrix <- pmax(adjacency_matrix, t(adjacency_matrix)) # force the matrix to be symmetric if by design it is not.
     return(adjacency_matrix)
}
# Calculate adjacency matrix for the toy dataset.
W <- adjacency_matrix(X=X, nn=2)
# Draw a heatmap to visualise possible clusters based on the edges created.
heatmap(W,
        distfun = function(m) {dist(m, method="binary")},  # because of the data distribution and number of points, it is obvious that it is necessary to sort points according to their symmetry.
        symm=TRUE, 
        revC=TRUE 
)
# Plot again with the created edges.
plot(1, 
     type="n", 
     xlab="x_1", 
     ylab="x_2", 
     xlim=c(1,7.8), 
     ylim=c(-0.5,8),
     main="Eleven Toy Dataset Adjacency Graph 3-NN (Undirected Edges)"
)
edges <- which(W==1, arr.ind=TRUE)
segments(x0=X[edges[,1], 1], 
         y0=X[edges[,1], 2],
         x1=X[edges[,2], 1], 
         y1=X[edges[,2], 2],
         col="black", 
         lwd=1
)
points(X[,1], 
       X[,2], 
       pch=21,
       col="black",
       bg="white",
       cex=2.7
)
text(X[,1], 
     X[,2], 
     pch=as.character(1:11),
     cex=0.9
)

# It is possible to see that there are two fully separable clusters.
# Next step is to create a degree matrix to algebraically represent connectivity strength for each node.
D <- diag(colSums(W))

# The Laplacian matrix will combine both adjacency and degree matrix.
# The diagonal is the number of edges that each node has and the non-diagonal elements show with what other vertex they share that link.
L <- D - W

# Eigen-decomposition to identify underlying patterns within the graph.
f <- eigen(L, symmetric=TRUE)
# Plot the eigenvalues (because of the dataset size it is possible to plot all the values).
plot(rev(f$values), 
     pch=19, 
     ylab="Eigenvalues",
     main="Eleven Toy Dataset Eigenvalues",
     cex.lab=1.2,
     cex.main=1.5
)

# Graph the spectral decomposition of the first 4 eigenvectors.
par(mfrow=c(2,2), oma=c(1, 1, 3, 1)) # set a 4x2 graph grouping for this.
plot(f$vectors[, length(f$values)], 
     type="l", 
     ylab="Eigenvector values", 
     ylim=c(-0.7,0.7),
     cex.lab=1.2
)
plot(f$vectors[, length(f$values)-1], 
     type="l", 
     ylab="Eigenvector values",
     ylim=c(-0.7,0.7),
     cex.lab=1.2
)
plot(f$vectors[, length(f$values)-2], 
     type="l", 
     ylab="Eigenvector values",
     ylim=c(-0.7,0.7),
     cex.lab=1.2
)
plot(f$vectors[, length(f$values)-3], 
     type="l", 
     ylab="Eigenvector values",
     ylim=c(-0.7,0.7),
     cex.lab=1.2
)
mtext(paste("Eleven Toy Dataset Eigenvectors (First 4 Lowest Eigenvalues)"), 
      outer=TRUE, 
      cex=1.5, 
      font=2
)
par(mfrow=c(1,1), oma=c(1, 5, 1, 1)) # reset graph structure.

# It is possible to confirm that there are 3 clusters, so it is necessary to save the firts three eigenvectors.
spectral_embedding <- f$vectors[, c(length(f$values)-2, length(f$values)-1, length(f$values))]

# Separate cluster based on the embedded features.
spectral_clusters <- kmeans(spectral_embedding, centers=3, nstart=20)
# Plot clustered X.
plot(1, 
     type="n", 
     xlab="x_1", 
     ylab="x_2", 
     xlim=c(1,7.8), 
     ylim=c(-0.5,8),
     main="Eleven Toy Dataset Spectral Clustering (K=3)"
)
cluster_color <- spectral_clusters$cluster
segments(x0=X[edges[,1], 1], 
         y0=X[edges[,1], 2],
         x1=X[edges[,2], 1], 
         y1=X[edges[,2], 2],
         col='black', 
         lwd=1
)
points(X[,1], 
       X[,2], 
       pch=21,
       cex=3,
       col=cluster_color + 1,
       bg='white'

)
text(X[,1], 
     X[,2], 
     pch=as.character(1:11),
     col=cluster_color +1
)

# ----- Creating a 2D DataFrame With Non-convex Groups -----
# Create a Toy dataset with non-linear cluster structure to test spectral clustering performance.

# 3 circles with the same center but different radius.
n <- 200
r <- c(rep(1, n), rep(2, n), rep(3, n)) + runif(n*3, -0.1, 0.1)
theta <- runif(n)*2*pi
x1 <- r*cos(theta)
x2 <- r*sin(theta)
X <- cbind(x1, x2)
# Plot the graph.
plot(X,
     pch=19,
     xlab="x_1",
     ylab="x_2",
     main="Circles Toy Dataset"
)

# ----- K-means (Non-convex) -----
# Here I use the R built-in 'kmeans' function to show how a linear method cannot handle the non-convex clustering.

# Apply K-Means to the created dataset.
kmeans_cluster <- kmeans(X, centers = 3)
# Plot the cluster and the centroids.
plot(X, 
     col=kmeans_cluster$cluster + 1,
     pch=19,
     xlab="x_1",
     ylab="x_2",
     main="Circles Toy Dataset with K-means (3 Clusters)"
)
points(kmeans_cluster$centers, 
       col=kmeans_cluster$cluster,
       pch=4,
)

# ----- Spectral Clustering (Non-convex) -----
# Since K-Means failed now it can be compared to spectral clustering, which should have no problem clustering as 3 circles.
# Using the same pipeline to perform spectral clustering on the new dataset.

# Adjacency matrix.
W <- adjacency_matrix(X=X, nn=7) # I use a higher number of neighbours because of the dataset size.
# Graph the heatmap to analyse patterns.
# Because of the data distribution and number of points, it is not necessary to sort points.
heatmap(W,
        Rowv=NA,
        Colv=NA,  
        symm=TRUE, 
        revC=TRUE
)
# Plot the similarity graph.
plot(X, 
     pch=19,          
     col=1,            
     xlab="x_1", 
     ylab = "x_2",
     main="Similarity Graph"
)
edges <- which(W==1, arr.ind=TRUE)
segments(x0=X[edges[,1], 1], 
         y0=X[edges[,1], 2],
         x1=X[edges[,2], 1], 
         y1=X[edges[,2], 2],
         col="gray70", 
         lwd=1
)

# Degree matrix.
D <- diag(colSums(W))

#Laplacian matrix.
L <- D - W

# eigen-decomposition
f <- eigen(L, symmetric=TRUE)
plot(rev(f$values)[1:10], 
     pch=19, 
     ylab="eigen-values"
)
# Plot the first 4 eigenvectors.
par(mfrow=c(2,2))
plot(f$vectors[, length(f$values)], 
     type="l", 
     ylab="eigenvector values", 
     main="Spectral Embedding Smallest Eigenvalue",
     ylim=c(-0.11,0.1)
)
plot(f$vectors[, length(f$values)-1], 
     type="l", 
     ylab="eigenvector values",
     main="Spectral Embedding 2nd Smallest Eigenvalue",
     ylim=c(-0.11,0.1)
)
plot(f$vectors[, length(f$values)-2], 
     type="l", 
     ylab="eigenvector values",
     main="Spectral Embedding 3rd Smallest Eigenvalue",
     ylim=c(-0.11,0.1)
)
plot(f$vectors[, length(f$values)-3], 
     type="l", 
     ylab="eigenvector values",
     main="Spectral Embedding 4th Smallest Eigenvalue",
     ylim=c(-0.11,0.1)
)
par(mfrow=c(1,1))

# Feature embedding.
spectral_embedding <- f$vectors[, c(length(f$values)-2, length(f$values)-1, length(f$values))]
# Plot clustered X.
spectral_clusters <- kmeans(spectral_embedding, centers=3, nstart=20)
cluster_color <- spectral_clusters$cluster
plot(X, 
     xlab="x_1", 
     ylab="x_2", 
     col=cluster_color +1,
     pch=19
)
# This method efficently separated the 3 clusters.

# ----- Spectral Clustering (Non-convex) with Gaussian Distribution and Laplacian Normalisation -----
# This section investigates further methods on how to apply spectral clustering, since datasets may not always have pefectly separable clusters or consistent sparcity.

# Adjacency matrix with Gaussian kernel.
# This method relays on Gaussian distribution to discriminate what points are the nearest, but for starters all points are connected with each other.
sigma <- 0.15 # hyperparameter that controls the weight of the area around the kernel.
W <- exp(-as.matrix(dist(X/(2*sigma)))^2) # this is also called a proximity matrix because of the formula's nature.

# Graph the heatmap, which is pretty much identical to the other method because of the data structure (easy to cluster).
heatmap(W,
        Rowv=NA,
        Colv=NA,  
        symm=TRUE, 
        revC=TRUE
)
# Plot the dataset and resulting edges.
plot(X, 
     pch=19,          
     col=1,            
     xlab="x_1", 
     ylab = "x_2",
     main="Similarity Graph"
)
edges <- which(W>0, arr.ind=TRUE)
edge_color <- rgb(red=0, blue=0, green=0, alpha=W[edges])
segments(x0=X[edges[,1], 1], 
         y0=X[edges[,1], 2],
         x1=X[edges[,2], 1], 
         y1=X[edges[,2], 2],
         col=edge_color, 
         lwd=1
)
points(X, 
     pch=19,          
     col=1,            
     xlab="x_1", 
     ylab = "x_2",
     main="Similarity Graph"
)

# Normalised degree matrix to adress problems with data sparsity in higher dimensions (the effects on 2D datasets are expected to not be different from the normal degree matrix).
D <- diag(1/sqrt(colSums(W))) # formula to normalise the weight through all the diagonal.

# Normalised Laplacian matrix to be consistent with the previous normalisation.
I <- diag(nrow(W)) # identity matrix.
L <- I - D %*% W %*% D # formula for Laplacian normalisation.

# Eigen-decomposition and plot for the first 10 eigenvalues.
f <- eigen(L, symmetric=TRUE)
plot(rev(f$values)[1:10], 
     pch=19, 
     ylab="eigen-values"
)
# Plot the first 4 eigenvectors.
par(mfrow=c(2,2))
plot(f$vectors[, length(f$values)], 
     type="l", 
     ylab="eigenvector values", 
     ylim=c(-1,1)
)
plot(f$vectors[, length(f$values)-1], 
     type="l", 
     ylab="eigenvector values"
)
plot(f$vectors[, length(f$values)-2], 
     type="l", 
     ylab="eigenvector values"
)
plot(f$vectors[, length(f$values)-3], 
     type="l", 
     ylab="eigenvector values"
)

# Feature embedding for the first three vectors.
spectral_embedding <- f$vectors[, c(length(f$values)-2, length(f$values)-1, length(f$values))]
# Plot clustered X.
spectral_clusters <- kmeans(spectral_embedding, centers=3, nstart=20)
cluster_color <- spectral_clusters$cluster
plot(X, 
     xlab="x_1", 
     ylab="x_2", 
     col=cluster_color +1,
     pch=19
)

# ----- Spectral Clustering (Non-convex) - Hyperparameter Tunning -----
# This section investigates the method proposed by Zelnik-Manor and Perona for hyperparameter tunning. This tunes sigma (based on local proximity tunning) and number of clusters (based on eigenvector rotation).

# Create an empty square matrix to 
dist_matrix <- as.matrix(dist(X))
k <- 6 # the number of neighbours to take into consideration now is a new hyperparameter that has to be set for each local sigma value to be self-tuned.
sigma_i <- apply(dist_matrix, 1, function(row) sort(row)[k + 1]) # the sigma value will depend on relative proximity strength.
sigma_j <- t(sigma_i)
sigma_zp <- sigma_i %*% sigma_j # formula proposed by Zelnik-Manor and Perona.

# Adjacency matrix with Gaussian kernel.
W <- exp(-(dist_matrix^2)/sigma_zp) # this formula remains pretty much the same, but since sigma is now a matrix it does not need regularization and the calculation is element-wise.

# Normalised degree matrix.
D <- diag(1/sqrt(colSums(W)))

# Normalised Laplacian matrix.
I <- diag(nrow(W)) # identity matrix.
L <- I - D %*% W %*% D 

# Eigen-decomposition and plot of the first 10 eigenvalues.
f <- eigen(L, symmetric=TRUE)
plot(rev(f$values)[1:10], 
     pch=19, 
     ylab="eigen-values"
)
# Graph the first 4 eigenvectors.
par(mfrow=c(2,2))
plot(f$vectors[, length(f$values)-1], 
     type="l", 
     ylab="eigenvector values", 
)
plot(f$vectors[, length(f$values)-2], 
     type="l", 
     ylab="eigenvector values"
)
plot(f$vectors[, length(f$values)-3], 
     type="l", 
     ylab="eigenvector values"
) # the third eigenvector is very different and would suggest that there are only 2 clusters.
plot(f$vectors[, length(f$values)-4], 
     type="l", 
     ylab="eigenvector values"
)
par(mfrow=c(1,1))

# Tune number of clusters based on eigenvector rotation (proposed by Zelnik-Manor and Perona).
k_range <- 2:10 # setting to look for clusters within a range and reduce computational cost.
j_costs <- numeric(length(k_range)) # vector for the loss function of vector alignment.
# This method will perform vector rotation and store the cost of displacement for each eigenvector, then the minimum alignment cost while K increases will determine the number of clusters.
for (i in 1:length(k_range)) {
  k_test <- k_range[i] # number of clusters considered.
  C <- f$vectors[, (length(f$values) - k_test + 1) : length(f$values)] # cost vector.
  C_norm <- t(apply(C, 1, function(x) x / sqrt(sum(x^2)))) # normalise the results.
  C_rot <- varimax(C_norm)$loadings # perform the rotation.
  C_2 <- C_rot^2 # L2 type of regularisation for positive values.
  max_contributions <- apply(C_2, 1, max) # isolate the maximum contribution for each cost vector.
  j_costs[i] <- sum(1 - max_contributions)
}
# Plot the alignment cost as K increases.
plot(k_range, 
     j_costs, 
     type = "b", 
     pch = 19,
     main = "Alignment Cost vs K",
     xlab = "Clusters", 
     ylab = "Alignment Cost"
) # the graph identifies that there are 3 clusters.
# Save the K with minimum cost.
K <- k_range[which.min(j_costs)]

#Feature embedding.
spectral_embedding <- f$vectors[, (length(f$values) - K + 1) : length(f$values)]
# Plot clustered X.
spectral_clusters <- kmeans(spectral_embedding, centers=K, nstart=20)
cluster_color <- spectral_clusters$cluster
plot(X, 
     xlab="x_1", 
     ylab="x_2", 
     col=cluster_color + 1,
     pch=19
)

# ----- Spectral Clustering in High Dimensions (KNN) -----
# This section further investigates spectral clustering with KNN and normalisation approach in a more complex Toy dataset (higher dimensionality and complex data sparsity).
# Download "zipCode138.RData" and load it as R dataframe.
setwd("C:\\Users\\sebas\\OneDrive - The University of Nottingham\\University of Nottingham - MSc Data Science\\Statistical Machine Learning\\Projects\\sml_project1")
load("zipCode138.RData")

dist_matrix <- as.matrix(dist(train.X, method="euclidean")) # distance matrix using built-in 'dist; function.
n <- nrow(train.X)

# Adjacency matrix with knn.
nn <- 10 # number of near neighbours to take into consideration.
W <- matrix(0, n, n) # initial adjacency matrix.
for (i in 1:n) {
  neighbors <- order(dist_matrix[i, ])[2:(nn + 1)]
  W[i, neighbors] <- 1
}
W <- pmax(W, t(W)) # ensure symmetry.

# Normalised degree matrix because data is sparsed.
D <- diag(1 / sqrt(colSums(W)))

# Normalised Laplacian matrix.
I <- diag(n)
L <- I - D %*% W %*% D

# Eigen-decomposition for the first 10 eigenvalues.
f <- eigen(L, symmetric=TRUE)
plot(rev(f$values)[1:10], 
     pch=19, 
     ylab="eigen-values"
)
# Plot the first 4 eigenvectors.
par(mfrow=c(2,2))
plot(f$vectors[, length(f$values)-1], 
     type="l", 
     ylab="eigenvector values", 
)
plot(f$vectors[, length(f$values)-2], 
     type="l", 
     ylab="eigenvector values"
)
plot(f$vectors[, length(f$values)-3], 
     type="l", 
     ylab="eigenvector values"
)
plot(f$vectors[, length(f$values)-4], 
     type="l", 
     ylab="eigenvector values"
)
par(mfrow=c(1,1))

# Tune number of clusters using the same vector rotation approach.
k_range <- 2:10
j_costs <- numeric(length(k_range))
for (i in 1:length(k_range)) {
  k_test <- k_range[i]
  C <- f$vectors[, (length(f$values) - k_test + 1) : length(f$values)]
  C_norm <- t(apply(C, 1, function(x) x / sqrt(sum(x^2))))
  C_rot <- varimax(C_norm)$loadings
  C_2 <- C_rot^2
  max_contributions <- apply(C_2, 1, max)
  j_costs[i] <- sum(1 - max_contributions)
}
K <- k_range[which.min(j_costs)]
plot(k_range, 
     j_costs, 
     type = "b", 
     pch = 19,
     main = "Alignment Cost vs K",
     xlab = "Clusters", 
     ylab = "Alignment Cost"
)

# Feature embedding.
spectral_embedding <- f$vectors[, (length(f$values) - K + 1) : length(f$values)]
spectral_embedding_norm <- t(apply(spectral_embedding, 1, function(x) x / sqrt(sum(x^2)))) # because of the high dimensionality, this step now also uses normalisation to further separate the points in the lower dimension.

# Final clustering & visualisation.
# For vistalisation the silhouette of each row is graphed and separated based on its corresponding cluster.
spectral_clusters <- kmeans(spectral_embedding_norm, centers=K, nstart=50)
cluster_labels <- spectral_clusters$cluster
par(mfrow = c(1, K))
col_ramp <- grey.colors(255, start = 1, end = 0)
for (i in 1:K) {
  centroid_vector <- colMeans(train.X[cluster_labels == i, , drop=FALSE])
  digit_matrix <- matrix(centroid_vector, nrow = 16, ncol = 16, byrow = TRUE)
  image(t(digit_matrix)[,16:1], 
        col = col_ramp, 
        axes = FALSE,
        main = paste("Cluster", i))
}
par(mfrow = c(1, 1))

# ----- Spectral Clustering in High Dimensions (GD) -----
# This section further investigates the last dataset with Gaussian distribution and normalisation approach.

# Auto-tune local sigma using the same approach.
dist_matrix <- as.matrix(dist(train.X, method="euclidean"))
nn <- 15
sigma_i <- apply(dist_matrix, 1, function(row) sort(row)[nn + 1])
sigma_j <- t(sigma_i)
sigma_zp <- sigma_i %*% sigma_j

# Adjacency matrix with Gaussian kernel.
W <- exp(-(dist_matrix^2)/sigma_zp)

# Normalized degree matrix.
D <- diag(1/sqrt(colSums(W) + 1e-10))

# Normalized Laplacian matrix.
I <- diag(nrow(W)) # 
L <- I - D %*% W %*% D

# Eigen-decomposition and plot the first 10 eigenvalues.
f <- eigen(L, symmetric=TRUE)
plot(rev(f$values)[1:10], 
     pch=19, 
     ylab="eigen-values"
)
# Plot the eigenvectors.
par(mfrow=c(2,2))
plot(f$vectors[, length(f$values)-1], 
     type="l", 
     ylab="eigenvector values", 
)
plot(f$vectors[, length(f$values)-2], 
     type="l", 
     ylab="eigenvector values"
)
plot(f$vectors[, length(f$values)-3], 
     type="l", 
     ylab="eigenvector values"
)
plot(f$vectors[, length(f$values)-4], 
     type="l", 
     ylab="eigenvector values"
)
par(mfrow=c(1,1))

# Tune number of clusters with the same method being used.
k_range <- 2:10
j_costs <- numeric(length(k_range))
for (i in 1:length(k_range)) {
  k_test <- k_range[i]
  C <- f$vectors[, (length(f$values) - k_test + 1) : length(f$values)]
  C_norm <- t(apply(C, 1, function(x) x / sqrt(sum(x^2))))
  C_rot <- varimax(C_norm)$loadings
  C_2 <- C_rot^2
  max_contributions <- apply(C_2, 1, max)
  j_costs[i] <- sum(1 - max_contributions)
}
K <- k_range[which.min(j_costs)]
plot(k_range, 
     j_costs, 
     type = "b", 
     pch = 19,
     main = "Alignment Cost vs K",
     xlab = "Clusters", 
     ylab = "Alignment Cost"
)

# Feature embedding.
spectral_embedding <- f$vectors[, (length(f$values) - K + 1) : length(f$values)]
spectral_embedding_norm <- t(apply(spectral_embedding, 1, function(x) x / sqrt(sum(x^2))))

# Final clustering & visualisation.
spectral_clusters <- kmeans(spectral_embedding_norm, centers=K, nstart=50)
cluster_labels <- spectral_clusters$cluster
par(mfrow = c(1, K))
col_ramp <- grey.colors(255, start = 1, end = 0)
for (i in 1:K) {
  centroid_vector <- colMeans(train.X[cluster_labels == i, , drop=FALSE])
  digit_matrix <- matrix(centroid_vector, nrow = 16, ncol = 16, byrow = TRUE)
  image(t(digit_matrix)[,16:1], 
        col = col_ramp, 
        axes = FALSE,
        main = paste("Cluster", i))
}
par(mfrow = c(1, 1))
# This method fails to identify the three cluster, perhaps because there is a big similarity between the structure of 3s and 8s. Thus, combining them into 1 cluster.