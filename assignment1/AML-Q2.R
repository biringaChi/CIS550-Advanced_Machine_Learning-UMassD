# author = biringaChi
# email = biringachidera@gmail.com

# This file contains code for performing image compression and reconstruction 
# using Singular Value Decomposition (SVD)

suppressWarnings(library(magick))
suppressWarnings(library(imager))

loads_image <- function() {
  # loads the flower image file 
  file_path <- "image/flower.bmp"
  if(!is.null(file_path)) {
    image <- image_read(file_path, density = NULL, depth = NULL, strip = FALSE)
  } else {
    warning("Incorrect file path", call. = TRUE)
  }
}

plot_image <- function(data, img_type, x=NULL, y=NULL, t, col=c) {
  # Plots  image
  if(is.null(c(x, y))) {
    plot(data, img_type)
    title(main = img_type)
  }else {
    plot(data, col = "skyblue",  type = t, xlab = x, ylab = y)
    title(main = img_type)
  }
}

grayscale <- function(image) {
  # Converts image to grayscale
  grayscale_image <- image_convert(image, type = 'Grayscale')
}

grayscale_image <- grayscale(loads_image())

par(mfrow=c(1, 2))
plot_image(loads_image(), "Figure 1: Original Image")
plot_image(grayscale_image, "Figure 2: Greyscale Image")


process_svd <- function(grayscale_image) {
  # Extract object, convert to double and apply svd
  grayscale_double <- magick2cimg(grayscale_image)
  svd_img <- svd(grayscale_double)
}

svd_img <- process_svd(grayscale_image)

singular_values_rank <- function(svd_img) {
  # Rank of singular values 
  singular_values <- svd_img$d
  plot_image(singular_values, "Figure 3: Rank of all singular values", 
             x = "K Values", y ="Singular Values", t = "l")
}

singular_values_top10 <- function(svd_img) {
  # Top 10 Singular values 
  top10_sv <- svd_img$d[1:10]
  plot_image(top10_sv, "Figure 4: Rank of top 10 singular values", 
             x = "K Values", y ="Singular Values", t = "l")
}

par(mfrow=c(1, 2))
singular_values_rank(svd_img)
singular_values_top10(svd_img)

k_10 <- function(svd_img) {
  # for k = 10
  sv_10 <- svd_img$u[, 1:10] %*% diag(svd_img$d[1:10]) %*% t(svd_img$v[, 1:10])
  sv_10_cimg <- as.cimg(sv_10)
  plt <- plot(sv_10_cimg, main = "Figure 5: K = 10", axes=FALSE)
}


k_50 <- function(svd_img) {
  # for k = 50
  sv_50 <- svd_img$u[, 1:50] %*% diag(svd_img$d[1:50]) %*% t(svd_img$v[, 1:50])
  sv_50_cimg <- as.cimg(sv_50)
  plt <- plot(sv_50_cimg, main = "Figure 6: K = 50", axes=FALSE)
}

par(mfrow=c(1, 2))
k_10(svd_img)
k_50(svd_img)

k_100 <- function(svd_img) {
  # for k = 100
  sv_100 <- svd_img$u[, 1:100] %*% diag(svd_img$d[1:100]) %*% t(svd_img$v[, 1:100])
  sv_100_cimg <- as.cimg(sv_100)
  plt <- plot(sv_100_cimg, main = "Figure 7: K = 100", axes=FALSE)
}

k_100(svd_img)

k_200 <- function(svd_img) {
  # for k = 200
  sv_200 <- svd_img$u[, 1:200] %*% diag(svd_img$d[1:200]) %*% t(svd_img$v[, 1:200])
  sv_200_cimg <- as.cimg(sv_200)
  plt <- plot(sv_200_cimg, main = "Figure 8: K = 200", axes=FALSE)
}

k_200(svd_img)