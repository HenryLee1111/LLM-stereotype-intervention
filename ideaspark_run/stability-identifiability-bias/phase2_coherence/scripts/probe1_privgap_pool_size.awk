# Coherence-gate numeric dry-run for the PrivGap definition in IDEA_CARD.md:
#   PrivGap(theta*) = |dM(theta*)| - max_{r in R} |dM(theta_r)|
# Question: does the sign of PrivGap depend on R_n = |R| (an unbound parameter)?
# Model: dM(theta_r) ~ N(0,1) over isotropic random directions (null);
#        dM(theta*) = TRUE_EFFECT (fixed, a genuinely privileged direction).
function nrand(  u1,u2){ u1=rand(); if(u1<1e-12) u1=1e-12; u2=rand();
                         return sqrt(-2*log(u1))*cos(6.283185307179586*u2) }
BEGIN{
  srand(20260818); TRUE=2.0; TRIALS=3000;
  split("20 100 500 2000", RNS, " ");
  printf "true effect |dM(theta*)| = %.2f  (null: |dM(theta_r)|, half-normal)\n", TRUE;
  printf "%-8s %-14s %-14s %-14s %-10s\n","R_n","E[max|dM_r|]","E[PrivGap]","P(PrivGap>0)","E[p_rank]";
  for(k in RNS){
    R=RNS[k]+0; sm=0; sg=0; pos=0; sp=0;
    for(t=0;t<TRIALS;t++){
      mx=0; ge=0;
      for(i=0;i<R;i++){ v=nrand(); if(v<0) v=-v; if(v>mx) mx=v; if(v>=TRUE) ge++ }
      g=TRUE-mx; sm+=mx; sg+=g; if(g>0) pos++;
      sp += (1.0+ge)/(1.0+R);
    }
    printf "%-8d %-14.4f %-14.4f %-14.4f %-10.5f\n", R, sm/TRIALS, sg/TRIALS, pos/TRIALS, sp/TRIALS;
  }
}
