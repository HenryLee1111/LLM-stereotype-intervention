# Coherence-gate probe 2: is theta_hat a property of the representation, or of the ESTIMATOR?
# Two estimators in wide use on the SAME activations:
#   mass-mean (ITI, Li 2023):        theta_mm  = mu1 - mu0
#   LDA / whitened (Fisher):         theta_lda = Sigma^{-1}(mu1 - mu0)
# With Sigma = diag(s_i^2) both are closed-form, so cos(theta_mm, theta_lda) is exact.
# If that cosine is low, the two estimators disagree on "the direction" for the SAME data.
function nrand(  u1,u2){ u1=rand(); if(u1<1e-12) u1=1e-12; u2=rand();
                         return sqrt(-2*log(u1))*cos(6.283185307179586*u2) }
BEGIN{
  srand(20260818); D=896;               # Qwen2.5-0.5B hidden_dim, from the user's summary.txt
  printf "%-26s %-12s %-12s\n","activation anisotropy","cos(mm,lda)","angle(deg)";
  split("1.0 2.0 5.0 10.0", KAPPA, " ");  # spread of per-coordinate std devs
  for(k in KAPPA){
    kap=KAPPA[k]+0; num=0; a2=0; b2=0;
    for(i=0;i<D;i++){
      s = exp(log(kap)*(2.0*i/(D-1)-1.0));   # std dev sweeps 1/kap .. kap across coords
      d = nrand();                            # mean difference mu1-mu0, isotropic prior
      mm = d; lda = d/(s*s);
      num += mm*lda; a2 += mm*mm; b2 += lda*lda;
    }
    c = num/(sqrt(a2)*sqrt(b2));
    printf "%-26s %-12.4f %-12.2f\n", "kappa=" kap, c, atan2(sqrt(1-c*c),c)*180/3.141592653589793;
  }
}
