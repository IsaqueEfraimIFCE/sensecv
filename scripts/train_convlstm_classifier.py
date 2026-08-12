# -*- coding: utf-8 -*-
"""Build the SenseCV datasets and train sequence-based ConvLSTM classifiers.

This standalone ConvLSTM trainer bundles the same dataset builder and
clip-grouped train/validation/fixed-test split as the DINOv2 trainer, then
constructs chronological sequences inside each clip. Every trial model,
history, training summary, and accuracy report is retained.

Example:
  python train_convlstm_classifier.py --no-build \
      --datasets-root /kaggle/input/my-datasets \
      --out-root /kaggle/working/models_convlstm
"""
import argparse
import base64
import csv
import gc
import hashlib
import importlib.util
import json
import re
import shutil
import sys
import types
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


BUNDLED_SHARED_SHA256 = "52c298f7497932347f32ec6d6ecbc1fe206053e21b2f4719c187f320ec2579b4"
BUNDLED_SHARED_B85 = (
    'c-qXpSDWHGvoLx;zXF;tHZ;A+VUcsrnc0J}jR6yEW1Ad)`zJXOn(mqR-TOSxnTu(#rBbO>DpeE)oy}s?k?fdervK_T;>6~MPky|;'
    'y(KpEa;y_)l&(oJ0+sCOQu#{NCQwU?6sfUpjElHHiGmJiKr<ev8L7e6sUw%TAkY(<;)RbNKUy<dkl$0hj@U5cB_&8;d1}3(c>rmN'
    '6V!4^jcw(c`=2p>usNMJa!KL5ItJ(laH073A0obI8FqDe2ZPX@ZOXG74nPpNB`v;7nCvKeIupTkbW3(hlOMV`nBI!RMVeSrKSm=7'
    '6Kus2^pavI5d$-4Ke)qaq$4PS9#cA6)UjKN=fP(gPZ{Si0K<WCiH)cO6Jl8Lhc*G0(G~IObV+^aTA;&(-'
    'h+oCC5T`G2iTgYIF^S*gS{CFfY)sV*oM`~^n?Tb;COb$!d9cviXBtS4^AA700q-'
    '~e8nxPe}Cw7+(Ddy(d0=$)VEITMs6?<oz2FuHi=%k{{bD~KbuWmD;v+UqVAuCgzP^w!)?TWRtN3%T@Q96J~rO5`~pz*9|<!-'
    '85@EIA8gV?-'
    'If*wha2+F4|)YBti$;!hw}oZekZ^%Ubp&<6t?QeDXD&%;lgZ56Y5v_4|wpg5$UD+bS|)r`dOIWzYz|C`cXV^l=kHaVNL*n!7?r(#'
    'c7ri6zIAnyJ1AYUb-<g!8c2BOp~JA81N0mMr~LFUw^0|nav8&8wb3={rHirmTU1!zFNUD`BJ(PFQ;{3-'
    'On$W?dOk1y57z=(kU#_&X-'
    'c@29~T=T8((J1$zl_zmZ$pnA*}fMAWw8++dLo>b|gT9Y230+LcrZK&a(w=~BLu#+tdf)8mBzoVY9CA&Ifqg%i%P%Qf~pojxz+@e_'
    'dw>Bn8T+vjq4?7p$vNqQZ2f51Ub0xr9o@cNt%$~SodJmL2M$cV%rKS|p<(L0gYsf7#mO(GK`=Yx%CL>mo+ah>$5T-'
    '?5^q1oWdJ9U`6+cLr?);7LnKV+ga(L+`OCCi>65)nTqn07rZZYy+m`W?^Gl`h7bXQOF!IsYjSMeRjXl*7VlPj<9K$5uME=1SXq)*'
    '0^m0<Ffyafn2dJkgpqSQl-yDZR3VL*FOPQKn*bAigf2@+XCBI5|G8T&CQ-'
    'X&jo7U2D8N@0UXIXuU@ACDXvPplg+Jit{BoQ!nE@R!_$2B9suvfkoaFw0eBYB6D2p8LXUW9)deU8S2__k&`HQirB`jwV0iXp@VZ4'
    '*XK3{da$ZvZZ?_9I8?lr4oeoTcv3aX`Z~@Vi>{iZ*O+el8>EV1CUHEnCe6&`k~A^RKzC&KueWo5#(`7zNpc|UgVZ=}-'
    'xAl23eG%OkOt%Yz@-j*Z@acw+J#Jd7~2hr$*kNDEwRbm7!USK)x1Z)9c9m^$jM8iGvc`P<+2mc=HThJ-'
    'K%;q&X^;mav7<In`xitSaxnY<@_X)I$MWrWvJkqI36R3>2x0VpI5oUq-UMAP0jUulsfiI1rrvG28ZRmU+hJeu}xgqkyN3)56^h>E'
    '`sQ>QaLe4*Ey$c(xp}GL(@imwQlzs%Z<U^zGkZypHo;fyDWa*xEN!3j}C|Y=;ZOZQ<jEn8ySoejTASu7N(Aog-BN7wXwKd?;2xtU'
    'Txt%ac4A+3*?M1r=uR$t7i&3hss4~`ygK0=;On)Y01YqQ$JS<cKE8<)gRmPWxBsDd-'
    'S77*c@KXBJK!_nR=wfY;Iu<Q!?8nsFkT^Z_lFZG-*m_J)`)}Uabg)OfZOE7qM2{-FB9ICuJzYzC^ZR-oI$2Xf)y&J7SZyE4Xi0EY'
    '((`B8FE3R3*UN<+60C)65tjtPcK72`OVSYO!}l%$8^}wjSo_<Q^IJHdAz;x_Vnqk}1<GtkZ6H9F=3hUORSUL5l56>DG#MRN5>(ud'
    'R-'
    'S;~_Qat3%9CazLWhYm&o4&aj)z=Ssd<CUTh{J@v#+@SIjTEZ)fk$1T)q+g6bHzB61_T<bozNMkFm#wRo0R;V88_fBY91#0okqk2Q'
    'v+AXFlguiggCk^u*dFZ<A&F-'
    'eR9(zi=YjZftZtS~JZ=dyMV@cLtReKM2K5=jC_pFIQDmG|tyQ`X$3>pI?Z+%~$1<D0u2Tg1<HD8IxYUF%<u!c_g(*irp`)9;C+Ki'
    'CY(OLlbUq-E>y@u}yvBXw4g+aO&w$x77<Y6>bhO(<;Zo=hoGF+^;W0V_>r0u4-P`Z+v2tN+-'
    'oY=lr5yV&C<lOzzWNLRbV^izV@6&g%X)n6;UoXz(+#QX&kEQ&~jry#+>Um|H^yk8&=_4G?Yml!aiG?!MAcCEI`Oz|x8kH_>vuPIG'
    'g{HyjK8|iySk)RMOB2yv^yEc-!%_B4I_FR!ehruA2FBn^S^QW3!s0Ha>V9{<<6x%4lHZt*Wbq>cMED%HEGxL-'
    '<X493H{w%@K8rP>ec2RNQ-eEHVCU7VJC5?H{&wBKa^AJU<gS^?@U&~1dNKwCV~eqV#z9ccv7xEw^Ot2IYgxxJS0Q7aoBNHv7a4a%'
    'M=RKiZdK~Jnc676^>i=$gWrCfAm(v>5N2nMcFtbcv-'
    '+aHyKytTYkaghf?dqzybM{j>;546h83h3@>X)|j5xP2#*@FaGc?NaTquHy4!y6&Of2qr`egDT5!4s%<ccAE_mZ7PoXMTHurM7{rA'
    '_2`Suwr&CGYeMPLwBIBFt{TuxpnWZmggT6`dSL_^f8bu7=!bvhEapoRDCahJycKZ5o2{OlsMVul7Sa>B<#Iq`I=WHX-fAn5BwyaD'
    ')$>`r>?mx>9VMbgk>-MuQs%15}VGBpb%6Bii=*_xo1dSQu_<)*`!|WNU`g+Tu$LXV+t=)@L27Qn$-'
    'S`&s*ztY@x;oT28ab@!|0X}(s5@)=yyT_cv-ml6Lpk?s4u2ZOzdEpzSaa2!0H%$R<h*bVz)9-'
    'QO0X53d!>~g}|T_0PDRJwJr^;&xDJk6W23`P4UwWI5NvAfiXlrmY7Nq0%ZqFOuB4To`*%M!jQW8$*WKGC|=Qq!3!5D*B*^2BeW^}'
    'RlL=pcp&PA%PhMW1I^ejpa(9u#ealUVYo$JqEfd_G$dqq`liThf_%N*O9?+520eVK_lK(6%w)iR^=2+=<Yuk!Lx-'
    '6fb@{;}7}8Sub(ur`ts8v}|^4_I>FvU4>64q8&&lk-UmGwnNrM%h}Eamn^-(Su~OCGR;-'
    'dQJ>?cZLZ75);{YZ7~t0{gH=01W{yEC<X^SMNedyELWd;I4_duYd*V-25c*ijgqD0=pe~rjvgxsgo-'
    'ZgMVVB#p=_M!5elKf0RGIUse@+|CWwNt6Z%*l>v1BX-'
    'sOfB;@>ukQLmA51s%#i*Kfn-J(=g{rJ-2Xqs(y;{V1bIQ&~_!(ZGumFy)n-uTKAqjLZ-bAlT2ysL+eOp+8K^SlU3~0&7*0$Qi`$k'
    'sdx1o15Qn<IwnQ;rZ{rg(b>px+I2FeYdh;5EuvzsS!0@BQ_wR!l$~X@Lvky!ZL2jUPMVu)-'
    '0jZH+?HXgHu8tf#43go{$+KanimkH+4dac#Y1zSY_8YmT&__Gq^!AS!#NU2e4KN;^cG{(i#n_h{NUdSgGjjPzZ9FCX^5(eBQ6Fue'
    'UKm7@?$iGGH1~`uF|IB*ihRPGfp;XQjyBx!}=g?5IMuL)Qy^~A(7_$k$kK)xHk9W3B9)0-'
    'Tizl849`&K8q14SSF=xL0I*ji5^o7Qr>ZK)bskVeSaQl@s{d7R5_hj)P+dzX4QbRI2j&V%1{$~x)M3<PR>@$;@q6No64YMXOUiJz'
    'sjTbfZ(8y$4VhkWX!9JeM2^4+z{D`dH2|rGq<B&Uo}_L$God+z`a|UdqaJgoW@f#{A6w=l6id!*hT8f;KX!$Y3GZ@7(2Z-'
    'IC2xr7DG|dyNblS?5@Y3T~}|SK0U4HMDmn5631yRfbzix1y;tVPoksqv30UUM^&rH2I5A)dt%LTqf|PVE)<FO+KrFZIj#B7IH9js'
    'hBK>smf&JlDtGRU0@Ngy7}e75n19|K<eWiQEAMw#9h0Rs=qVNg=YpZ7cdcUGjDETwx2sJ7A8xr=tm|tA?M-'
    'zWoxWomi<(Y$U#BNl=Beb~yNvo>#&IaMq5PI)(ypluy(*F!_J~-qTc23aRzE`|gkm=v?QTmE(-}=9@LYWDAMR*J-'
    '=8TTy?iQPG@QVa_k^f%7LIHPyDK5&nn5<{2o1gKy@o?*u<NdKg=S}ucz3H-Io7AUhhj28hXOlC+-'
    '~1_EDgpapmyGpp!LOWk2j|#v8E6kufs;4lOF4Xgg0ZK_3h36*|<1eL@`=z^aHFp<|v+$m0fnF40WW+7k!}Hq%wP<+wC7c{W<G5dq'
    ')xW=sUZ@3}!7+>v1kSLnnCO?;!B7O~tZZnCPcy-<xr9Tf3#|F*+RV#hnfI%gKO8OeNyhq)@LNPhwp#N0Ls;i6mxtW!uG^-'
    'GshdPIVK^z_8#Jo^ZHVh+L=XYP@Eu<`;%cF27mZYgKbRt(p(3EOSZ}lQpJW<<V$KALoLhwxM&Nw#iZ0HDZQ?THfn$ZvqrCbnM;rU'
    'Or4F>{mm8(UhC5Scn`7#TrQ#gX3VLb+RI%8nqH>dNwsqeN7KxUCq2glHjk$$g*hiS}n*apXmjeg+L|B`^Kr*@0AQ~^GqE2*JFKqN'
    'H}A=)s@W7SLx(wYQ)CMP@`<qbhOV(*lEUkMUUs&_D6HEZQf@XBzoK>f)2=1ezhB0f&zhL7pp=5SYM=T)$^n(ba})Nx@C$IBdcC*U'
    '7hU<n_{K2MnxPIe7I>ItD2m3(yb1KdJd`6bj-'
    '8{0&d;s!bY;<KaUG*<GSP|TjuR3(;g&g!zqK5n=QnZ3%N>Dl_X`Pw<ZI}cGb?zE&fh;Hu2VqWV2Kuk0<|Oo9~wl@k#kWkWPGxxqN'
    'h>mdYIBrdH3}92R1JyWUP=<sN}q+=s#<vS{(0u!;8yD3Q;%JRI8!x)YQ#lxJv7g)NP<IT%H4LVUjThU(S)#Lu<%Krl~|?JC=COp?'
    'jtPVWgE&S+>`KF8K|CbQi~+o$vtJ@*`XF4c(E#;zU_QfHvIhr~s39_|?MU6aH-'
    'WYwI$`i7J9$aowWrm+-N=Vtp}XP4_vT#bEU-9N-5L?#+WEhK6e+XvfbR-+mYF@Pe|ctt-YOM~t)O(nWX%Ql`-'
    'hdD21NqZ0Tl|1@IEmz;p2i;m=7&qHf`h{k+#wS^GC*1LvCXKD7hBLTl(QdlxV~<b_o~&lv*NloDbWj+K`-SMlL*$L&mAxl8)9L-'
    'LTG?UFt+6_q&4JKLOda)N65+$mC6#hzkA-T<XY}HugL=+T+<2-'
    '#L<7}R*jLHekmWQw5@$6_f70?6+rt#?nC?6AvttkiX0Cqg7$YN}*o`Nzg#I{8puu7+s6Pip+lt6X`&ODNf#Y_J?bDG-cvw;=bUGl'
    '|K7=J=hkl$42M0uA<e!|!!_zVv7<(hc-'
    's;R=d1FSj7`Iz%Ga>F>UUvzt>95X7t`pQ365BY}j1o0^L={#Mx3Rrkx2u^*qt;vGsA%48ne~*RV$&l!(r+Rb%s84b=Ole-'
    '3xI{(lk_<oC>#5w&JxQt-L$8iSbK(3*57J6CbcE+Tbb7t#1IuS;r$#1XBbhm(~-j}6zozyn$wOZk5O-'
    '&m(trqsJOBr4u&A~Wu_2CaHl<-U-'
    'r!wa{{lLG{M$#ppnUXMzT45gh}?#_3ggWFqrDTRixACt<v7~q&6w=i(P*u9tj7R>vdA=47D4F8e4NO$+0xq=$CsL^f()0J`0WyT`'
    'kKn=-`N6&}*Tqh{nFNbU>VE2BRuR?5junF`SIHR%yl>h$Be1?+JwRr$ED1>eD8(C+ZEFeXCKr?ez^UJzLpdz^`I2Qw}?pJ^S8Z-'
    'd<F&yRpYzH8(D2<D}_k#GJ5Tf<TV(t^LA9#08IY>{(&SIojEi(afo4aN8-3QeJ8|a0iH8EVUxs)*{I@vhCua?k-'
    '%*dIS~f$8sS=#eGpd8mxu7g3o1kvvc>pLoDsXD(;KA{1$|YEz*HPR5KbZHN<mDo$x%1S0;G4wlpuYi-UtgQm51bG0Z8VhM?l5={z'
    'CT9@h^aEJ%$jJBcLPb*GKha}}{W>=}$9IdtLfX6ZwvV#CjPLK6ejs1peaU?#KtvK>akgYe94UPXG#?8wxul7(WpU9>n0BiiLM^Vd'
    'Ww5+NApxZSZ1C%89ga;(^$8}lRP$Y_8hCS03w48-'
    '1mBgk5RpDXWJoGe>aYmy)Mrh9$8Rtx2}hH}N+pwpI2Y&i<Gn=BKWuw8<yw@v*;oGEvER<tlS6@Ap^gxdI-'
    'V;u&TODSrKXv*oy7#3pDtAkEW4#%czYG}1^H-(Ly)v|FUUR&5aGG|PUTvPDIEtSZibHLMv*!~m@>n-'
    'sz<;qk2R30;#CisCO)9&C|7@f@f{XU-_v&l0Z9)wKFP~Jd);jVZ?<UAVDuZ#MVEyt$aIqMcVICke<!ksr*+3o>53CT)z5|5T}i*='
    'Vh8%(|4sl+=i9%`auZO^3eTuGngk2Qbj8eJH+S^wNxJ+%9tu4v_=eY>xbZqf4$U#_m*rCG?|_A<3?&K0s}DxniQ?(cg2Sj<>Cq`H'
    'PFlDlR$Z13cXq1+)+9v3c!gkC-PJ#k*J7N@m{F%}{Dag>||ihcncb1_qR;|}o5EZRz@)53mJu2(B@E9HvUMwM7LOt_0ROS6&Lm#V'
    'bWmNJ#-{j3~5#Mk4!wXIHMnp(zpC)AzHVhEC@#}0?sC%I5cn`yxj9!0BY6)*-'
    'RnA@&R5JQLt8|fRH)PQnTulsp`5=`k)J+rq@$>P!|cpbPGubukjHd{D#d`RmOt`>IsWZ_)Eu<^o;&I9}Gu%g-'
    'iEzfkk8wZQ~^&(&=Mu8Q=(#?a@)yPyv=&aX`<HgW%A4*5@3B%>)o4{pQEOi<0mh4V#nbm>G&v=U$4`nY?v^hG(>?LwR*HbZXZ!dD'
    'QVsGRkhRRTlc0T}Iwt*uWs|pjQ!tfp&oAozc&gOKLE`(#bsIU^q%^C;1>hhP{rbHg0ioRvM6`WbTO{$t*oad*bK$U!&*Ta-KmWPK'
    '^VqDi-)8l?U?nv0XY<APhxYB1<-'
    '!x6D)v+n8kL@hY?#bm?F_Utk;dNWsT9jK<^%=B0*Q`6##ir^9W?u9&S6^l%xJe8xui^~8_jtlJtdc=Sad$JwH!|M1e}|4@#aTd%?'
    'GwJtc0)&y<!Q2sln#P*F$}h>%%w)pGI3!@$5-'
    'BVFq@h>hDBaA*38Sp$Zif2^{xS3?!uL3YL$r$b`3pK?0Il{cOA?t)#El3Sr}TpIa4^`^I|yDwl*84Muv*+#w3M?@Sfg~3~xDex$P'
    'kHvjgKXM+>9o_3H6oiK(dC33*o!_sy`@YB#%-'
    't)7KFsV5nFp1of3r_jZg<Dzr@u)6noi$b#+XoYIFU<Co%VcSo6&YNy?nO&IpcBK-uwivMwq=<I1Wf35kEA`ZQx@3FK;|d{M31qlg'
    '*{72fu;Mwwu!uExWW#kldBfbOckwvqy}>$FqI!E2(|gxPW6ame9-'
    'R|IpxAAj7K52(ds;Q@m33fO8XWuH(^46#+8<wg7Jpgnr;bse#AK+<nJ8@+y~uVLh;DN=ug{zqI=uR6`kdIB8a|6Pvy4;6tJ#q7xD'
    'A0M?ee?&WoM{pZID5_*~GGg5PKb^o1?PJ-rI@QbU=0a21otcyjN^s!o|}Ig=Xg-'
    'MWbx5yE_D0_DZAPtr_`zh)t?wFqBLJEo^F~9Sdv@?|LVuS<@OXMkcKH7&XV-ES`1*qn_NE?I6uKQ#_JWXCr2(2lGI=-rrTLA*Z_-'
    '_UA$q?|xxS0%6nQ81dw(iP25eg`_W&jWU$z8;2bNr|$-<U7noT-5sk>Gu$qv#_@U`2_Z0A{xHcHV-0I-'
    'e<}BaQP!H^#>P^s#HL2q%EsS4QeA6pdFDOQ;DO_EL@of%=~Xz9*<bm>b-'
    'mMPef@pdQ#PRto3JBOgRp9+C>N%h+ZJZZ&{3?mJCA1A(u-'
    '{JnZb0u4yKBm#?aXdreanDh&{2uGZ0A@E=59&Rj+Kb!dQ=OSK8hpYep8^)VEg@746aGfC=QX1#FW^b>jwqZ0*<@`^r!pk7Z^iqRf'
    '@?2Di)4^hCp02-JI2z!;7=*0e3PF*Krn?-'
    'r=Hn|Z6)znsg+xYG8H43}tbVJj3CBh7&&FxJ@4#m)IryN?`>QE!4DIm(3@P*nqo6U$60TfBpjnf00^iM%;g+W8E2e|S3f1j5XG1h'
    'mLWgfBXSr$~G!5QNbrEPAJDQtWlx7GieTL_B+CC<krMW!>5Op>T5MG6&;$R(2R-u61F-'
    '*aUHW^$q5u(6Jj&jiZfGB2ea9(^8@y?(frm^lIJg7qzJ9imupcIq1fUB4#)q-'
    'G*pXG;x)XXU9y#)ruN98RCvwta7UOhevagvsC)Ij<MJVKE6~0wVs>4)C#wrBPinuO@{87+2^e1^jN=J&su6(dpd8O1P&W}5a-'
    '2Y2W3tT(pb*TLc2uR<UgOg!)qtd%G;HpY{wOF%NAlCAycxN9CI;xt_C|hC$-'
    'vS8aZFJzdFv#i~4H6ZYB%^%aAS2E~#;;y<W6T)y1R}E_9E1;OHH9yBBM{O&%)SWM|~;x%$r0i85T+;yS6G^R#OnNCyIwS~Rz%T~o'
    '_?Qf?Wz88&m|x%1hP-'
    'B(9x6TT|NL~b}whoT`pF=L#rpog?|4GSy3MGT3mX*^3i63p0T_xkB5a+p)AiaFl(<x_E|S+mZKmGYt+>hrPFzS0zPY!*-'
    '@ncDEnEps}%TcuV$zDcA4>x?s)Ik--'
    'ky}#{wldHPP-D%AYZpwwLpjb!05Z|pw;7l#~^%#L3GdX9YvW%`rdSQM^Ao|QeZ_dTanOQBjYk82)ur)oq-NKdKoLf|NnyTeOQERD'
    'Z9Pj$|xp^`qi^j@)w)N*uv9M{fC)?_QI?5550_Cmy4_St4Sk1=bWO0dh*5$p~=MItnaf@q|_8q*s>9sS_DtF%7nfPUTHX`%5nG#C'
    'b;~>`~MyMs2XX{40sbCqA=5(m-'
    'G!AIS+{0H6hwAdWX0!3r%v(W;(9)MR&AE!F><Bu<c&*X!;mPPaO1rSA<=A4&=R&&aGg<Y`*)?Ntt*?!1wiYY1<zTDo%JBVNDd#s3'
    'r)BVnq{-T;SQ5^|Pzsw*Evi-oawKB;MEKk<PL7sMk!>x<+anH4#BMefJvQp6ac^xQNk`L>wUn-'
    'd+PTs>(0fOuVXE4@xTcjCy`91^T#eI_I&k3SM|E;BDda8f=^^d#SC_SE#$Mt)bjX!;r4Gk_!^;jV(Y~q0M)U|b_j-j|dVNlldW-q'
    'sFyN!o74siAr|9|YbH{?2Hn6E&svBfO^a(l>tOAFPy?Q;y^n0pqK8tzO2*TG5yX)GGbXS{p)`_;Sl}IR@K-'
    '<aCirMjG(A2DreTPaShtR3^F}$p<s)MQCyDH9S8!~DPCV>sVuLup*!)-I`*<`IMu9ma1;V_vWK-'
    '}>{uc)l6Je{tET=LT7XL@%aRy*bkR@b181R5=F)s7$W`I@=z0`_6tmmE@a7vVgUS@x2crViUNVZtVs;WALkOb%)^8TZ=5yP2yw8j'
    'tL|F1I5JtjkPn=BsY4h7^z8xn)c;;$)GkwS_K@6Jf!f%i!fi97!QFre9rVBDv|5@w!@`<$BZau%k7G6I!^hNBYB?rZJ5Ny2)X-'
    'R<=%O?7%!V<C_ef%+0*1^YBobUHNu!<(XUZ{qWVt&MPHf_RLrp<LM~obYlLTtKrBws>kcLU*pT(b1Cfa(3zs<L>R@mrC+d)eS4Cv'
    'Rhx6ZXHQ*=)~rRGmoR$23nyZgj;&H)_HJi57Z|#)WHmFQ><i*RTW8j2D3{(OJbPy)m)W6#d_QTcmBSgt>an}*-'
    'QYCgZgc9{>u44&mt341Sa7F_Fa|wbUf&^39vUIjM!go7jl{zJFfrvW`JumWO`ZmW^c2Y&x&^>tK)aedv!VKow|gw+R7b29E}PV{a'
    'r7UPUeec##?94{e`jh`{H5H|2ttbf7#BQ+&kmO(>v0JBaIb#0pStP^l=NRSyrY`O%#?4IFtj<BF|bNd3)F4gVewt<ns&tkAa$mTM'
    '7-'
    'f06K!`Fvwb)VvOBVljBk5Is6}FyBS&#OLXRr_%&2aO0asDh;_>g@he|u^^QvswHr33D`H3(3j(RKZC=Ptp7U523qYRtjjmPm7YaF'
    'puw9*+CbA#Q`W{bGzCwdta^%&W6h0<X!Vb2!=)`l1L*3<r0eO+!kef6W3m2pCcR=#iIg2Og(HW{spg1yb9gv4l+M6%Ub^6WU3Ets'
    'EX#|tE>9~Je>Zf<ot7nY_K)-'
    '_aGd#k8tKulb08tOHC6^tMBTrbyHxud4Abr!9;*2jkGlrCWI5Eil2h0r?g4e{PN*BT@OXb}ibaj}dnEY$(Nin}>05oCi_6I;?&7c'
    'aXS`8iUt_$QVMb*UQizy)4sj-'
    '9%1x1ObH>?rQPFl*YJ>7i65M0AD|wy{o@mXlPaWVBJH_OO0rLP?mm(4Qq1=fco7am~XgLy5<2LJdT2RkhwKWSq0r)@p2sS&S`Hjv'
    '+6c_-;Czyt?V(t`)Iib-YS%I;3TnpM%)k%)GLZ6J{ca-'
    'f2&>c`Cm8iEx_iqpCg{=!N>0LgJLjOcDWSdh1X1TfL4Y;&UWgezxkKAj?9!*v;F;R^6;mwX;p4U9y%63(lWL$0I|rCRXW&n+akgv'
    't>{talJZ92FbH}>wc2O7yd}r6`Sg-'
    '>#WgiKTPezWj|*~4OhFUVZNC~JiU$2cqoj>eXDMdr+V1|w+N#1#&ExgtSIl2bByCVPa_>{no4CyT03_=1jFMvUo6`=zeQLEm~m}w'
    '9aee)ch=DepaFO3n5j0plOawI2iwuMT~9P3cp^2o8T|T7zPujBu-Rdn&krivVn!XFm5W49rd?#Y?-'
    'h2{oyl2@x3k+_@!*f_j|a!7b(x@KWQeD0{YfTu@$O<f9;u-'
    '*E^FqB3^)zCW%phXPu+B`=E4`rsWoP;gfp>@d%bQI8&y?n*mq~g0y0Fi7qr}pj<BMuz>tY)V&m@v{WaKVw(CTf3{hAEY}})Hx%BR'
    '$d4xM{%;lUhHn8Stq@RtMOU+5GW+|2zXp!I9*Qjff?hlXWWGbIkb04D1X&7W~<aQ;ZA4?<`)oZ)HL?fmBac$o=7M*imf2|MMZm8q'
    '6W7$a2kRqeR)>)1<(Jh|3MvCQ{F_Wtz=OI{#a6jAGGvtn%PfV$E-'
    's?PWEzPVNdrpi>(F^dJB`lM`^<r}#X0*Y?b)EQVAKd4sd3z;#3a(QYkGV66(+LruCWK+6+eMi8tWbz8#daWKONG~_#?>Epj|qo!!'
    'bESLizVKqD%>j7Uy|z*Vj5%dScn;qyH=OE7}y@BUEkiTFO(f2`dp=FS3g=ZBvWl<TJ`nv{)G#7bs{ZbTwKx=Mjj2}yn)Cj4Qq3{o'
    'hu=HG}JJJl%e>b{Td+BBOz}q3_=4}I?=d@)TT=`NAxKk=@t*E5#EX|XSm%KsxSIeBi|H`Sx!u6It_Me>9?#V?ApYtysf(!Vw<TCS'
    'yaS$vRQ~U@TIvAWQLj`VVvyj12nwhI+Iy$w9^#SBQdOYmKNvE*X53xY0Ed{JF7NHo0|1x(pkW`K%VqO3|3!9^phiZXP>*CN}Xtqt'
    '7Z*+DYfM$<4`BL8ze_)c@$-'
    'ln#)+#+jHC1I#F3#IXA!L)r}O~l)QM<=mn|{me<FM%XPoac)Iy*TiDxbJv~kgoQAA7nw}&`YnpWy#Lc!Gar&8JG;a$9_xcja=TJV'
    'e?fDT`#!oke#R0hZFt5N&W=zh;rB%D7I3pImcy+kdn`?R`kqo#-'
    '>!@Q4LNS)(n!N_Hl~J#7T}1F)Eg$KJfr)IeHO_}@sS`ri!%cN-XbUdSAlF|wJV)cK-RtbNph?NSYo>xBLtfmw_iQi_X=MZ0b`J6>'
    '8a+cEIg0jUBSod#yl`kn>qOGOnSdk|5;ta=89VLmWw|L*rzd#}*(sh+2JINycE)tevK`ZSqIO9Yj);SBV0?%muf%0p<t^Do&AAP2'
    '^n;yPo31SNeY;r<E*6_XKGm>ODGy7vd-Y?6vd&tKd1P<udRb$oN!LTuGw?RfMoZ^FJ-'
    ')ZDINlUfBU}1nw)b+t^ORE2=+Zdb_6MWgOXa#23EaL4?pB?Rm}n2}RI{ibvx{t!vm2|KVbNk7rIO^J$gpK5H#H$Wv%kx^NJpe>GU'
    'q}j3hNmbgQYbDLMG{zeV;vs8DGnX?9^PTZoHBUW`+#WA<jg*=EicseBF*ZB~vmUIuwA8AL|$rZNWH-'
    '?pnr@keDq}Hd=4n8;JNg#qP!#qoLyJH%Jxv5otxN`;zzAwDgN(Yf<mcc<OO#wJ;uQ)TDmSv)Gc)kJi_7#25ha#Gc-'
    '1STt%*!q_U$2&R9f_GIANCt|^RUJp_RQ-<>5icuR&aT*g4gJlqKs%{EBX5l6dt2i};xEW&U2RY2jhwA2jxiF5!GemtCEFdw$V-'
    'It7YesM)#v#2QuYy;x+Dl$SCp;DKhg~IWKha6er=>g-'
    'XpB60^Sa6}4Mlq}7I)OP%23;xJLGEAfYmo=qYY%nPyL+1<Pdw*+~T{I){QcMF;V6QND>dljwXF4UTvhE>5UhtOW2y{=A~1VZ@BWA'
    '&N8kYL80q(eYGb}`8Jnb66<vETJkxKtKpQ6<fEpnd7d}x#j4$~VC;H4YvIOS&AA`tT|~`$I183|v(sg&#<fsQiFA<<$2LV1es-ii'
    '8DFWWHytZ<x|OU^MmEx6b_PqS9_yHBwtW~jTjC&+EV|jL_i#OQ+?CeY?D6T(u1Hfnfec(#_nlA2eUJ`<X6t4=;7D*ZK~dIGAdsWO'
    '3qvmM=;cD2bI>{BF44LLTUNSBg3y>D=*1IQ>x6-b>;TN)KZ;n4@pic%-'
    'gVLSF?w;MB@3`42Gf|Gi8}n_#d3+P61Hi4rti@nGqPPq@V46#K59EZecmRVg>0p4>D4D`pe@p4I4^9^9)L+aE;u&TT?4UkiAd1lC'
    '4}P{p8)^ge38ZXgOJNXw^M_mN8qEgd_xPt7R=StC?K@CLNyg!43G$jK_Be;E7Iq&^TK5<_>QCFrW6X9$I*4c6k$r1kwx_E%=|DPw'
    'X6g_=wWEva&5B{eKL4FPYWZx=6(QaY_*^jq`d@dZ4gGBO*mh!7B`weXlQ!v(HVi;nT?J*V)fzF-ZjcZ^k#ascBIzD>zLcy{OeAmJ'
    '~x77lFYDGYZ;G+{t%mO@(nNJwwkqQg5!8TC0apZ8)lm6Sb+BCd@d~SGmj^pMODPfL!;`)^dW2KblHfV{-'
    '!!y6q9B9AbC!hn6!hnWo(NW8Y>TlmDjI%)Ho5_o=?k!sWA<8;$vqgW?TYlO!imGTJ#O{)5v~48eyK(l|WLhRCeo2ZY=bfO$WAH#}'
    '?#G$3-sRn$I#><E4b4EzhZx@-d-'
    ')?&MC3o3$^xzNiN2uoX8jp2N~TT*T+Rkez4F=FT~5niZnH!JyRNw+NTl3{J?airmCX2_CXZ`22^!JRdyMM|*$LTZWq$hMMYDZgH4'
    'LwwTeGZ;XdddxcrgkLJv{Ss5S5!?o!1btaCAIhk))&V4qzC#i-'
    '&cSc0Ypm#(v2K*2~H?+F_ZT6{QLuD7lDb<;lf<E47;^zB>Kbm!x=Tn1;y`~TOg_Uj}qU%mMkSw)Nw%BaqNN3mCQ@x$0#~!ee^9_%'
    ')0JatS;|C~n&|ym!A7dQNQA<#ngP{M)WBvS@SE?sKA&XQv!+-'
    '(=om31VkUY(a!awginjx1PSTh9cU?_rR7dndB(mczoC`J@M02m3J$RGYyN=kHU4aKXPh+7+g7%gD8ik%yVulk`k=<pv;i5s5zegs'
    'e;@&U?KHlPaWgJU`5WsSzqcU^^LC{*eNjs%Jmb!Dkjg%vZPT4bE&d6tL8AK>l3_vnM~9}K>tbowvqKYl*pjMJb#<_`PEg~1@@u?g'
    'spfQ6G+?h69Rv$`#0oRsipndL!R&HFMd9k<*Fy6<YOv^7+#oG{_ABG{#%m4V3V?y$8qFHEf>0>t|{p_wtJ4v3K(`RSMfm1b~CfIi'
    '^C4GK?zDN6+|n1l>Quka})^^smlkie{kEqo+Zdtt-@gEe6HU?g$_uEgXbFsXktnM;?`P52Si*ktlqr2~3T^V@``l!%KWuMiV-'
    'O2u38X1evMQ0IX~;J<KK_hnU&)(BRp;oy`3RZb7XK#oCm6Tngu?|@pmXR@OT)~|N&J(I}G4?jzg!n)6uNm2_EphANsqu_%6`u#&;'
    '-=CNS2g80sBtYzqz?TAYx6f|<K5&ksm@xv2q7bzk`oU8IyW9dyK0xUgkOsPcG;GkPkv~9n4}@?dKZNx`?#2ju*X{lF>z|4|`uSy3'
    '<mXfM+E4Vmwj>Lxz!>pGBNWgb{xv9dembytI60Gqoo-'
    '2QL6IP}WOq6o@DC_+#9<L1y91y8QFj$AB>RwH$QWT#@f{%jBc9UJpQyST_iaCDSarwWuf>CSeNEa22s~U*L|cs;?<yj$+LXlFBTN'
    'od7ykGx%^#%Igysbi(?-'
    '{Zc)!EwL6|l`!5wfdKMlw6G$SHEe*uDRxZihmn|CJ>IB>u|0TMU}N+{t8q?Tv?e83v!t^I9T=}+|atQ5$1hvdWV`}|s-'
    'iNJ1n67D=$bNB&e-'
    'iia8LO|r|#}C5qkp&_0s>)7Tl`%bq^`l|69`Zw&;h?$?k%6NhGion4K{RGjx`1=%i_*Zm#e;{=!y+I*6AB%lh@+d}G<*zdJAp(Hs'
    'VPs32OT-1$b~{TS%v{rrGm>3`J8Nl&tIIa_q{$;xdF=-L+Vm~Q8pQom-'
    'y}k|D{2J+=gMl0T}s_!fX_YK=MP$C{ynJdEj?gV@Q!d(YrUW{u78Ym9UWI{_4>EPe}{b7MuuBo4V(8227t+g@D44%B1eu@2~cpjF'
    '+@pVEmh)zaLS13XdPHFBJbbKriG!=%;<Y3H^Ij=n<BT!;`!Mlckvrb%QRqQjZo5!UJ5m9=D;v!rY&HUxF4miUc@678HNLr?=x1P?'
    'nS`ZogCuUt=V}26f(0BhUdV!tWn&mB1n^kN<$3wuzqblb_2NLZ}G*4}GQ0{*nfT9PqRY)X6IdeL!}{<o7E3K2!Ivx&4`$zFD|e%D'
    '0B{voiIPMo)VIK2p~UDltF*hXIxAx}P6&mSzwEXjkyim9;AP>5dE0r(HlVp-_8JZ2yPJGp`OHL_p>9=U-'
    '(wt&8=qFu14I1Kl6?B+zy#pu;ON;Il~4I!=Cq;&o8#FCB|C)GqIj&66q=E+4u!2iO0TCGgK}0zV%PNmWSgXn?qQcZtJ0<Zyn&$Pe'
    'KNvUvz0-vLu^jevI_2(-xZ2OYdn-~(JIfrIoSN+Td3+-(`H&#MB@T$TU83O$>xKVa!SRO`XJlQj67r0cx>t5X|&$PIoeFMi7^-'
    '5ao2q$z@y8KC1Tn%)~vy?P4P65I#z3aIi%g*xY`DYo9=5cOBhAKt5-'
    '?kyjn;=Stk+(hdNEiZ8f<nN4?u>RchvBg8j6~4zfikA$jFZ(_;QoAqd6%C0Dx2-'
    'k{+5W@eGJ{ud&sRsN4c?wQz_u(Q0F7FbbQA^miX?E~wJQ*S+g85tLhTGLsQ7ZAhdV0X9wrRmz%KG#73w>BHN^z_2p8DtbSn=5-'
    ';y(koXf5}+2Q<Z!(rMBkMf+rMREokI+X`od<m^w5<IhiIMinFMi6M_N~5TZMNx3o+%B~ZPXo4t7{Y}Biw!OXA*gq7#Rsr>!8TdQA'
    'H^Lje}%-'
    'n{P+QHUct)MRJzo>Z3JbG|LRU}vkd`ZDmR9xlh?h?Ch+kDCnYD&Kk|E<umUzJd_AUDa9V!%<y`|wd<ByD4kXllZxDT6|3$*P)yT*'
    '1QNEuv97Pi+Z)2Y2SZQC|Kb-b=Z84T`yErJD3p@e>5F8P}9+n=uD4Y9c<GpPd$XZ8Ar5gYQp-'
    '7$r3@GH%x;P&DXqU;rDbWnLw?&XRbiJV%@&M<9Z>G=+lz0?20`sIv_yS5W&<M`LD??Oicw5V_XXyH?8G?tNZ)fQE|6qnd;IW|j;}'
    'm@_rg+Dr|I2y$D4X}&c{)GO)A`Ljp(K;9&G|1Tcu}-'
    ')pyIF$O@f^+L81hNXLqW`4Gli%`v8m$p8k51PTaTOk|qEiNDdy*`l9LGEHCMbz!mlD4bBdOG7#dR3PvD#bpQ@E5#duF&IWz}IDl8'
    '^g9}m{JPd9Z5)ldXYfq{4>fX)}q-%;01%hht%GVdkN+@^*%_DeNBL__6H$ZngNKfB&3L$PT$cCb^Q9!Jbf}~1-l?T`CKqV0x@9-'
    '|82S>B&!UI<Y0Y!}&Yi>w+EI?3(`vA@}p2rW`_}~#RO}%bZX$i4;ZYg8<*pghyt&SAW3J3yJB}ARVyYBuQ_&SExZTPW2TsVM%ID6'
    'OK$O0GfNOMobx%Z&0etqinTW1aH95U-jM=cl^GS7`cfNn?a-'
    '!=Cj1#Tab98Jh0(Pcl)2JjD5x}aw&49#`~45~no{`|hFo)f6%ylDQf+Vn%%tboJK_6YYG07Fgjm*f5WM*x5fBrQPF0S){S`G1}tS'
    'fB7kpS(V|UGRNBS;Mw}*zbKmU^6fO102*Tzrx{51T;E*MquKAgFAJWJBnCTv;Y>jr&i_Z^T_h%e`@q%B#p;@srKUO0n$$m%Flm+p'
    'aRy;ZSZ%U8<qU>5VrPO{wlqDgfwwzZ6*$-'
    'udv~uKmjjav&tQe%e{2EA^8(0$qi7j4gkgkm;@q%r~)Fa?zoQ?g)<Mpey(tmXE6ySxfj^aQmbD;W7|-'
    '_AhxoQmcGM{rjhDHdIWsozaKyX|2Phgh%xpFBNg@oJ=B2|^8JT{0h4&Lfr|i?av)OoU@UU=A+k#vIsyS@H2~m^_E|Z)6k#s=E`f9'
    '<SQu4ffi0%QA7lvvu5T+rLy?_KXc8!+I;qJAQ1_I(^@MlXCd_T7Qz)r3K-'
    'GhhFdTCD5<>ofSHRu*T<)|}nc=>4Cu?I#&WcCUX*wl5oS+H$kE-BH<NT5|(%+I)l`YkAxCJ-'
    '@kmXGRi^2EzPUy7s2W}OBJ$Q*xk~<o@`5AVN|Eo)s(ssM!$vr)J`vu4SeMc9?MT(p>AQv<U=Lg9HaA5?n%0OHTZ|K*)d)bD*Ype!'
    'L;~SAJL%Zdn`D`K^Gl0BWgAetDR_2c=beRAR;m4zIMLJ+|nZyGhd@zjSzI{>`B#$)oJc2+12)$tlxV3O1IZUbLi>A`jgNqqKu;N4'
    'O@>)_uKp*NvR-'
    'b11&`ehM)OH_0Gh?AYYY8eu`5!Ow1(XEJ<yb*i4k~_uuc`oP_#hm(#&L<s8!@c({g5=<89g3T3<eZ4k%H%>{r8Vo+SMI-)zy-'
    'i@tz@`Lokm#Ir$Sy{>eS0?VX#KNb&;RctpLlQ)WMnr6tYgh9^*qKf#>94Hv%S`QSqr@J2!Z^4J%Z$5P)aV(G4{qSqs;kWyAmTYfc'
    '0tG)*`Rc+xrHzGI-l^&{N76xC{!0TOK*(Y$v7M})s9#Ccw&{bhUb=-'
    'Go`5bQFU*qv@tl(jxNa#CP*5{5s3cR<62!V72$S=?vNr4L+vY+B<fu1Aay;D3V@9=s~bvwMw)PUUZ@*xxylMN%OuHb|KteW}pAjf'
    'L?lG-5;ch2Km3C32|Qf^rx^7Mpz2`%TMyw0_diPt;`BDG5IX%TtLGhjz9l>_VjeiZp?*c!bALE1Nk{{S#~*Fr-KEVkzSddfJsC*D'
    'Iy2=@E;^Wh*YXb`((SVi2xWBqgo;_~(>+6bBAhbEP!fczK6rR?*Ds9_?D(c?YT+@$Y7m6FYbclrHm8$s5lW)oolLdyqC0y#a|loy'
    '>@vKlqhM_~`ZO7Bond6zD>mG9q`B;W(>9#S)IVhAEHv)hO28dJWiJ~8Dhbcdlo?&<kI%;{(NC(wg^_6zuYM@cDgaQRoITKJ@VxV+'
    '0VgaQcsQ3*bg;xZ6DIp83FmKGnH-N>6XM?eVaFWAcnbmP5Ala{_gRNv(3qv+>_*Xd3-@Bpn7G$3z-'
    'pcmcwlMtP}ouC1TC>PMeMgAqnt$#lF3huz5_Y9<P;_dZQ=>Cx>sTBvJ9QRntJ>Sq*r<cT)ax5A+VWoh!y!Yj=$Q!vC<8P9oBRz&+'
    'VK5Hg0&NH<U_?%~Jl;rdqYNRpT;U9GfMam2k*8Jv4%(f8duwtB3xkO@Pw=1u4whPk?f}I8;r4sLY5n2$c%?sfFI-'
    'sQPjm+0cj)?+{&?ih3=l`aQGb-CuF0p5iuxGQpewlzkhislkDZdh!@&v^^6O3k)Lu3y=zK<G<W*m^vLn>5_jbTt*B09XaTaiJfSG'
    '9cs@mS|5YT!cz(IcRj9@t_q3hQ@Hb7RgD;cM85;*ISmkQGpvfwJYB^3t53;>s&0~nj{KthA_L3-qo9)aA3-'
    '#<Rv$q*xGwyBOciEEzQ{rZo|_4{F)NTT3zcYvv@;mS|5&#^px1o20FnSp}rE>&hui(sb$nnB0@m*~T{_Io0$`8dBGcw))&1qt3Gi'
    '3v#*GazRqED!gtqy-3~`rt9Fih~_1`7f2%-{EC{ve5s6nLg)NKC?BLp~7>rWs#9m{1mGK2C{_@D7imUviQDZz!zqPoT-'
    '7yhlV!TyJ~)hD>t3+!9I{XNFVQa;cIZ8yDQ+~b^xY37(<Xef_L3f$$=<Y_bw+2iLCZfRJ+&CC{@FU^uO~-'
    'YS#1}%sVqpCJewpq04Izu7sgF`A@xp;z|ScNTaOeXL$w$j+pR`fU+;<ai2=&zNE)}uwcJP$JCqTp#F_Qy-d<{KATn2V-'
    '@yI0!fcQ07HXd1JK$U*~A<A@&u9^zCY+YzwW*1z}A;NE%(d!OZ9JR5>QTiK>miS-r;TXlL-'
    ')HQt%i3!awoE&?R|*hbbr<;17cWhLU23lE?~60!;e;w7nIal)vxr=6MXpI-!*Xq%8*%Rj3t+u!p}qJRcz{S&A&LbjYgTfx-Uzbbi'
    '>~N$nR0zVREp*w{CtONjwJ@8a)$2C@@CfdTkR{rI_&0$x*D0DdX?5u%R#xzp`;6z=<#q=+PP@4}L(atkN_{v|~<f4_gb`#=AC2rx'
    'YE+`sT!OeiSyC1c@%E{M|l$I~eb>>y|!0aqoWb_L4-'
    'l+PFbkUQAw#}6P%kheVsj|HII3nB}mGe8@De9y7$64>v6zA)3`^{#%RHFCdT8fq#*phhL{Us>^1z5v9OT`gH8kBS$J-N-'
    'LxOXcb9O)$V08)zOW<~-cc$^dsnQJhVnEdrf-tSF%*g=fHFsyu;Tf^sZK%BKsev7P&@J57`-'
    '%*A3;9ymFl7ZzG^(i$qF5B$ZnbB4Lc@(Yoxb(GS{-~-?yMr5@eRo~q``UD03a*C)q+Fo*$tcGi}0MRR16lXqwhg}db5IqD-'
    'dO&1Qx0wK{Y$Ufq_zk%F-_Q?uJ|XBAz#&`(U1-ZLUpNsZnycEC(iRvTK#Hht@y~s<wbA-XSbl(w4U%0D2l^!hIT8ve^HoUnF|v0b'
    '6umK&5)B2W`x_)4h%Vnz@*kaq!n^zoESJUd7@Jv8hkupfe}T`kod`ZiR_vP^g@6XV8ts6eKZPmY(97jhW}S@Ly8>R`DbQU6tZg+_'
    'UG*NpB>87!{J#CiY5}v=<r)A%^2CF!xCiva*oEwvS3C<K#voavQX#_r113}fja!=59!^U36G(Y6vVQ@2dw-'
    'L(FRih>zc$Q*0a^DFm|OSz0bn1=Vg}FU)Dx}nLG27nFj(=v*EcSw$y;0Q_K$2^+iWtPx|0G9_(aJW3p8Ze=X6ry>+;x|DKC@zF5s'
    '&bvVc9Z!g-of-rYr=RrH>wco$}#OhG5fy&w<w>(=Zy`pJFqKaA8$ce=HLP7L@<J0D~4OE&@n06sV${%Fkqy+Qw<8TFEhq^S}&S2m'
    '_)-oK1-'
    'JssQVqlAZ*GPgtN1|H1%mzJ{f8+|1<Z(%Vx1%+lN?7dU$xhx5Jw!ffj?Ow}OM!n^%{JxJHL3{rB7*v2f8~YNNhy0e`=%3USZ8oaf'
    'dE2YE{VEFtRW$sg<*k8_H~#tD=Wm6AwklE@jO0mCNBu`}*n=O(x_Q$;A}UNUeTTo{C90?M-'
    'H$mTP>!Rjxd8H<CT~v~7mTdZQ=obvayHV7Lm6NVCMC*gItmWzBLR7#Mrm(9lwvMHI`9E%l6CPn1Kj_%*MB({r0TOW2qm06PsuSZv'
    'B`5*)$bR2A*;k5WTEpZU?bzKg>8P>z5x9DbqMYg4Wtf#nD~D|0M)m57vXutT*|BBS8uPf(lrW#{;vW{pCEk-Fa0$dT9`=+D*Y`+a'
    'BV*q9DGS**zNzG#EL@ki<1X@y<?<wl}A{=A9p3lGRHVTSx~(rf;$9(tYCmfwFJh%%Uud$CNOBWyorI&k`|XsUOw&jC)Xf|$N`i5W'
    '(i79KH_71)kQQo5S8$f|4$6R`z*!qos?Jd7Jon4r7%w(nAOjCEO|_O1>B)PrKU=aDpOkmYX;wZ_Sb6jkz@AZx0Mv>00|Y)2LRGA-'
    '*sxPV;Fp1!4rlLgIrACBR&K;R^{wxXz+XzVA3low-INi_iOfosbqGLNQu1UXGK0K`q<x!s2gZ)7s;mE3nK@ZDL}tc?KqO<dH73=('
    '4MB4PxJfAG!vxXX1q7h?oa5~2>#z#X|m_{Vf66RgFkiG_(Dp7l+977&;vCDCYC;($p3|js1h1FRsRFiP|9jWb(U{8*5g#wbgwT_k'
    'QhSps=RKad3<E=o)q7N1d&}H_nY2vpGx<doWMfhzQIM}JI!X6uv6?{5a|4a!7G<W$-HWnlPy2luh8!=gF)qtx-ZRa+4}|f{m+&*b'
    'fah%A}rW^_MhCGHJ>}*#Fa#vJ6OMDBl!2{K8jYSHc*2#8UZy^8BPHGroU$MYA{p+{xyiLVxQkY{AW{utv{pj0OR)hJ5m5=q-'
    '0CLxc&{$jq{`f7L$G$SNLTIzDbZl|NE5-L^k_mk$sZ?SapueKPSmflBN9RMcR)VxrS2)A$A3^pYZF?-@j~*-'
    '~WENC*O5cNf|!fPhSD>H=QSrr><IBr`-MR<$3aAy>y|Qrtn%)Yp>rQ&%PJH-'
    'd`LTH@PC<pgQ*Sr969+a>=B^W8e;w53t)K7>|?nNXf1OEuL)IuQd>uqoq|neecAU(<i@%^vTeM6=ToO-eGlx<nVgaGBdt`c~^M#7'
    '|s2<iyD|z>orbqQ1C;lNzr&fNmx(?{=a+!<W__7Xy0Gn@9fUcYy?AiUogszJG4yx7fg89EE1N-){0Dif1`OzyY&t1eQ=ix*l-'
    '*t{a!anG6IYU>ulkV1WV^ENT=e#&sRf({|)^T9szQk7B^M<re@m}H_1b#o9d3x%5(s)KWYYGW_MCU7kL5XDd?z;_-'
    'eR2SjyP<^*~^BIXd@PH4c89hPg$2r1at2Qi5g0Ijy=oEbByC7!HGX<b<Dq%9~K~B;D=A+fcvm_w1d7eK;oJ7tLq<H|M1k<j}fH-'
    '@fdB7Z7>jp?(xLYOy8Y_-'
    'C)}yW+iperU~iBW)yj#!L_LCVAXUFE%;Ynaa&R;Nen^62k!JU+(e8`pj?T+y73c{k0s^+;&PS608jazrPOI>+K2TU1jHER@-'
    'AUIok?~EIHRqrjwd)CZ(InkK~jyIpa)e1hdY*5@;67EgSDi>OB;`-T5cw;S|xKG8|}f&%KX;Zq1cuGN&nM-hp^v;m>qZTPWS1*y^'
    'dk;j!_r&+k7iu3Gf)-1GM0_vO)<dh@d%{-'
    'bU4G_}uL=&`{~7)i$ye<z)u7FMY|mTq8J0$<5k;ejW8__>#)tK2QB5?i{dFi)FEh6ZN_1)K5}{c_d+ib?*S74875im$R(kL>jV{^'
    '#~D8S1vo;MyioR4!E--j^cErH8k(17x!7z`Y1kMhgQ9LO>Y829tp$mr<cvt$f@;Y{AP9R`>#hW+;%za*!8Zi6y|8wrxzo3#tfM1G'
    '1X|3yWkq8r}=VNO$Rg;Lyj51MJt^Gf%$#mNN+-'
    '9{zZtUEYAaR$`C`C%R0n4;bG8QP>^0)rZ1F5DtBG9^Hk1)a?%QYZx)?gpM5nIM6U$AZc1TJ>f%;5)Uv9ziq^c&Hwh*_zJ)M2M9c}'
    'sfNdYFy=EA|AWDwyZ(s|4{*N5Pl^29hvu~4@>fy{L<67Nf5{z?7ZqjUQ?ksI1UXpX77$WI{+Ho8DfS6&A_5GEC}9<_0Vo*$#*|Cw'
    'pp|_WrF@40>+lj^5o27p|M%|R6_p_S-'
    '@jhT9`IC)YTt?eKli#B*SB7_KX+3kE&4DiZH2&{JNNUHW62<F1~2XbyZzuqUPj0siE|&&dcO~;X-'
    '^+V2YRR4pMcYO^+G<3AZt9#>y{7jfWi}a1yQLg6uy2a-dCFep(`(K_z~7!$^EAT?<Fxl(co_HeA=-0&>kK3-'
    '+}&O+ux1z@|06GhgY0&kK^N!r}lby{QLD-pAWLnV}2g%>t2$2_4}gTBdI2G$oEM~94k2jLF`379HqK@o))Ci>$_XP{?Pf~ACJkSC'
    'r$fNO@}{l$>kov$A*(GFgwZKozvn1H2^uS1sF-U@n0{6HY4;4pc2u(gZj8se}pG{*uDd(Sx&Ej-'
    'Ds{l?*RDc6XCX(rG5C4h+lS*<ib~b|GQH~5@P?+(0>N<x5)e<tKebO|9@#R|94u2M!x?8?E>TsEXUAQDW$MM-ASqr2IsDTEbeQI?'
    'qq<(%gWs(%9$tW$@h}`j(tIK9GIq*5&pJ&`%Ey9s7&o(22A+vP0AC8cZJ<#heRK^-'
    '~i4?3Lw~(9!M7%zRUPa%0gJ43i{MhrtgfB@?St^=KWLXdO&WN%3beg@n@d7TP2m1|4WYfFBF0Y<@klwe+T7JkA5hf)AW_oqwv)ch'
    'daNaPTGyOj6-LO=uS7tKh+cY=jE}|R0XC~naad_I+T!Gk*TsPZnq26K=2<REoDtesNNy-'
    '50u3^6_^}UR;(+zT3JR)y<b2b<x_IvePQR_xu)3g@0u~M+WoM`PT9Sm&NwaK?p+a6cb%T#+?1!!U=-'
    'h=mdvDed3C5gs@n2Ruut6{Pp#*Y4z0B5bIjqhPvbL(J-'
    'RbqyuUl)pYXb!9xu8*=!8!l@7}%Y`a*A#Jq53o`<J?%dhRIwUYQcF`h8BGRRDZ`@e0y|)~0wX{(!W+xHoj=F5luSb$NxU-'
    '1m>DUKG<!+eA>hJZ<a06qs_k%YP*-fdTqh$9g+}zQXD582U_=|KeQwkIp9Pbd~C+?7yGl$FGj5FB?cF)?E$P>+<JXx?z)i-MV-'
    '(*#<Yu_raa>wQ8ev-|X#=RqAgQRu-'
    'ht;l)#`aQ*mHn*M;w>8Bzte*PPB`Y*cEPiX^#Zw~1j*!VyHjkq48x3C+OdSKDO%Yf8N5%CY-'
    'GY;wNU9I-rHEExVt)*MlG<<$52uTlRq<+g-y(qXwep-GiZ1(G?Dy%@cwR+|1rzt4?r1md_5Ev@r|8SbV?)0$#|FZdANLz0HbhbB?'
    'lyT_!1DCq8^2x1ik)N-&?azyJ;yx|TcfH`xEx>W*b6!2_;MKqz9wzVttr8|tNQt-9o|82U>59^at0fdhytRg#Sgq8M_U?--oE2T;'
    'mMHspL4}-QEr}c$00Wn|JwVrR1?^OsC0*+viL6hO=jk3^&_}9fQf|P-)GN7xu+V?=BM&d@Gaoa36;eDB>t(-'
    '(j7phH<qOho=sjKu3xZx+Ga%tkmrrH82;`jC-'
    'IX#$Ig1PQwxd!C(IcLpgezpHZ(yoA=L6Kpi!6!n;cOVy2Y-WPbI1t3hNTsS%G8iS-fqCXb-'
    '%?6wYOg)_XO_YQruS%?;AY)^Z}y$%EG&@1FsfRKEwO+m5&lv!*D}N`jr6b=~p{L-'
    'kQM70XNIRvDGbv!}~qJ1L@62oS$xBMnDaI;}JP9A>A?*2AAElB)A3t&nc`h)f<P!<#HG;tU+(!yVMP0j#_dVwJE9~^MQ}Y5I+eE<'
    'jrQ2uVQ{zbM(VcS2Tawk6~@{^~^xRsAXhe(^JyjZ!h3!mpp$BQoZo`FCnVUzXqt>2>mYo9zs|O$2<d3FOB{Z2D~QPrd|{M9i+$mq'
    '`!nGZ7BI-ZyUTm`ZJ(U7fgQ*3$VBCemhCnNt^Bk*r(g4zlRDojSXxXo3g9Ep7x82slS9QUse4bG`Yoh)RQl&wn-OGzoOm!&DVd1^'
    '9_h^dV{0-'
    'D+IL5u>V9~DQV&z)}(>_7a^>aquu3DpkL8i>iIQzkknz5l03envz7_+H5}z`?eCx{O}>U9XA}JmsK*PvzlJDZ@ckVmxy9EIfZ}MA'
    'ZUp~+yJ#1Oe+>z$hj2fCi;k9%@{O2+H;;cSq+o+@@Q^(nFUD3}AmAD%a2A9?v8Mcc2ujhTdbf`9yK&%{Vh*4ri#vep3>d&;QmqI~'
    '`v<N@pd~8bp+&Hg+%&hJkWf$q$DW*z43LyhLaFppJ`aEWR+KK~SL&dW|8O^(Tw5r~V~`H&Q0omGx{z|3-'
    '%D8d&|$aV4r4GRCEweol(Fa(P$0X{A?XhHk{sk>h&z~UEx!Vg-X8H#Hwe@-'
    '!EHpZ#`%16pIShpyX`{Q`OyrtgULPIzXd1z3jU2%b*l|9%m_-'
    'u+Ad@LgtL<KcSk$BM!Sdor9FL42>eV$3BuDxmLDlw^}!(tfhwBNN-'
    '8J$S7QuVhIywPDQ>}=!%)`O?!~uqGqm&*7>|t}&PP<6|I>>biz!Oqy$TJ>AYeYe<5m_WJ^zq?nCr~~m~Hw^+NMNwurF7bd@J|{gr'
    '}wb!ATiOz!cg+C&O0&;7{^MSV%R0(zxnxiYb=^q5rS5E7oZkh@pGG;sgs>q5@ktBp8q?Kw^aGhDuBbP`3*8-?4r6*}gbw=-'
    '4*CUVP5Q_iR54stR5hL8p8)BwX5dr>nKC++I)lYMpI}`!xa}-stNgAu-txQm3kG*RcX%{157t7aTlGN@sZHbj?=yPP+8IYJGY%RG'
    'K3@bODo}pJPQ!f$8=Wk0gguxeArzSc?0Lsyf!W+dSc_te{yU8CMr4WjIxefRwx&q7`Ta)vT2(qL(#mhH;hjR%i%rf&WA3;ZeO-'
    '&(zojQLJK2Fkrc6@}MtG&dSV3<IxYNM;Yu6hcdIJ`T%;>6(m$q2aVRyqy#DpyP^y{%gs3P*3fP$K0?ZGm^UpfP^4QroPvzkVfvV@'
    '+X3=fko!?on3F<gz<i9=ZsKtA31I!7NfYNLPx$bJlc-LIO97qC{?v`V#*CI0r(XrKXrl|lzc#42ERSpgCRcd)Zo5tQ&rc6XIEh%a'
    'gxkc7<M3Q#jv{2=s%v<qVmn@Ltw&UHIpYg9=Zj*f`yHir)e6qb^!g2Z&C3DWJbb0w{`w|8@&_hEMW$@XhNPZR1th@qNPSFu`$V7b'
    '7|!ei`(3yC1Fw+`7y'
)


def _load_shared_trainer():
    """Load an external shared trainer when present, otherwise the bundled copy."""
    try:
        import train_dinov2_classifier
        return train_dinov2_classifier
    except ModuleNotFoundError as exc:
        if exc.name != "train_dinov2_classifier":
            raise

    filename = "train_dinov2_classifier.py"
    candidates = []
    if "__file__" in globals():
        candidates.append(Path(__file__).resolve().parent / filename)
    candidates.extend((
        Path.cwd() / filename,
        Path.cwd() / "scripts" / filename,
        Path("/kaggle/working") / filename,
        Path("/content") / filename,
    ))
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        try:
            candidates.extend(kaggle_input.rglob(filename))
        except OSError:
            pass

    checked = []
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in checked:
            continue
        checked.append(candidate)
        if not candidate.is_file():
            continue
        spec = importlib.util.spec_from_file_location(
            "train_dinov2_classifier", candidate)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        print(f"[setup] shared trainer loaded from {candidate}")
        return module

    print("[setup] external shared trainer not found; using bundled dataset "
          "builder and reporting utilities")
    source = zlib.decompress(base64.b85decode(BUNDLED_SHARED_B85)).decode("utf-8")
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if digest != BUNDLED_SHARED_SHA256:
        raise RuntimeError("Bundled shared trainer failed its integrity check")
    module = types.ModuleType("train_dinov2_classifier")
    module.__file__ = "<bundled train_dinov2_classifier.py>"
    sys.modules[module.__name__] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    required = ("CONFIG", "canonical_classes", "report_from_predictions",
                "pipeline", "dataset_is_current")
    if not all(hasattr(module, name) for name in required):
        raise RuntimeError("Bundled shared trainer is incompatible")
    return module


shared = _load_shared_trainer()


CONFIG = {
    "input": shared.CONFIG["input"],
    "no_build": False,
    "rebuild": False,
    "stride": shared.CONFIG["stride"],
    "jpeg_quality": shared.CONFIG["jpeg_quality"],
    "day22_clear_dir": shared.CONFIG["day22_clear_dir"],
    "dino_threshold": shared.CONFIG["dino_threshold"],
    "dino_max_per_clip": shared.CONFIG["dino_max_per_clip"],
    "limit": None,
    "datasets_root": "/kaggle/working/datasets",
    "datasets": "obstacle_freepath_dino,deviation_lrn_dino",
    "out_root": "/kaggle/working/models_convlstm",
    "img_size": 112,
    "sequence_length": 8,
    "sequence_stride": 4,
    "epochs": 30,
    "batch": 8,
    "val_frac": 0.15,
    "test_day": 22,
    "seed": 43,
    "trials": "baseline,deep,regularized",
}

TRIALS = {
    "baseline": {
        "encoder_channels": 64, "hidden_channels": [64], "kernel_size": 3,
        "dropout": 0.20, "lr": 1e-3, "weight_decay": 1e-4,
        "weighting": "inverse_frequency", "augmentation": "basic",
        "patience": 7,
    },
    "deep": {
        "encoder_channels": 64, "hidden_channels": [64, 96], "kernel_size": 3,
        "dropout": 0.25, "lr": 7e-4, "weight_decay": 2e-4,
        "weighting": "inverse_frequency", "augmentation": "basic",
        "patience": 8,
    },
    "regularized": {
        "encoder_channels": 64, "hidden_channels": [48, 64], "kernel_size": 3,
        "dropout": 0.35, "lr": 5e-4, "weight_decay": 5e-4,
        "weighting": "clip_class", "augmentation": "strong",
        "patience": 9,
    },
}

_FRAME_RE = re.compile(r"__f(\d+)", re.IGNORECASE)


def read_sequence_rows(dataset_dir):
    """Read labels while preserving the temporal frame index."""
    rows = []
    with (dataset_dir / "labels.csv").open(encoding="utf-8") as file:
        for raw in csv.DictReader(file):
            path = dataset_dir / raw["filename"]
            if not path.is_file():
                continue
            match = _FRAME_RE.search(raw["filename"])
            try:
                frame_index = int(raw.get("frame_index") or "")
            except ValueError:
                frame_index = int(match.group(1)) if match else 0
            rows.append({
                "path": str(path),
                "label": raw["label"],
                "clip": raw.get("source_clip") or path.stem,
                "frame_index": frame_index,
            })
    return rows


def make_sequences(rows, length, step):
    """Create padded sliding windows without ever crossing clip boundaries."""
    by_clip = defaultdict(list)
    for row in rows:
        by_clip[row["clip"]].append(row)
    sequences = []
    for clip, clip_rows in sorted(by_clip.items()):
        clip_rows.sort(key=lambda row: (row["frame_index"], row["path"]))
        starts = list(range(0, max(1, len(clip_rows) - length + 1), step))
        final_start = max(0, len(clip_rows) - length)
        if final_start not in starts:
            starts.append(final_start)
        for start in starts:
            window = clip_rows[start:start + length]
            if not window:
                continue
            padded = window + [window[-1]] * (length - len(window))
            label = Counter(row["label"] for row in window).most_common(1)[0][0]
            sequences.append({
                "paths": [row["path"] for row in padded],
                "label": label,
                "clip": clip,
                "start_frame": window[0]["frame_index"],
                "end_frame": window[-1]["frame_index"],
                "valid_frames": len(window),
            })
    return sequences


def build_model(torch, nn, n_classes, cfg):
    """Return a compact spatial encoder + stacked ConvLSTM classifier."""

    class ConvLSTMCell(nn.Module):
        def __init__(self, input_channels, hidden_channels, kernel_size):
            super().__init__()
            padding = kernel_size // 2
            self.hidden_channels = hidden_channels
            self.gates = nn.Conv2d(
                input_channels + hidden_channels, 4 * hidden_channels,
                kernel_size, padding=padding)

        def forward(self, value, state):
            hidden, cell = state
            gates = self.gates(torch.cat((value, hidden), dim=1))
            input_gate, forget_gate, output_gate, candidate = gates.chunk(4, 1)
            input_gate = torch.sigmoid(input_gate)
            forget_gate = torch.sigmoid(forget_gate)
            output_gate = torch.sigmoid(output_gate)
            candidate = torch.tanh(candidate)
            cell = forget_gate * cell + input_gate * candidate
            hidden = output_gate * torch.tanh(cell)
            return hidden, cell

        def initial_state(self, value):
            shape = (value.shape[0], self.hidden_channels,
                     value.shape[-2], value.shape[-1])
            zero = value.new_zeros(shape)
            return zero, zero.clone()

    class ConvLSTMClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            width = cfg["encoder_channels"]
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 32, 5, stride=2, padding=2, bias=False),
                nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(32, width, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(width), nn.ReLU(inplace=True),
                nn.Conv2d(width, width, 3, padding=1, bias=False),
                nn.BatchNorm2d(width), nn.ReLU(inplace=True),
            )
            channels = [width] + list(cfg["hidden_channels"])
            self.cells = nn.ModuleList([
                ConvLSTMCell(channels[index], channels[index + 1],
                             cfg["kernel_size"])
                for index in range(len(cfg["hidden_channels"]))
            ])
            self.dropout = nn.Dropout(cfg["dropout"])
            self.classifier = nn.Linear(cfg["hidden_channels"][-1], n_classes)

        def forward(self, sequences):
            # sequences: batch, time, RGB, height, width
            batch, steps, channels, height, width = sequences.shape
            encoded = self.encoder(
                sequences.reshape(batch * steps, channels, height, width))
            encoded = encoded.reshape(batch, steps, *encoded.shape[1:])
            states = None
            for time_index in range(steps):
                value = encoded[:, time_index]
                if states is None:
                    states = [cell.initial_state(value) for cell in self.cells]
                new_states = []
                for index, cell in enumerate(self.cells):
                    hidden, cell_state = cell(value, states[index])
                    new_states.append((hidden, cell_state))
                    value = hidden
                states = new_states
            pooled = states[-1][0].mean(dim=(-2, -1))
            return self.classifier(self.dropout(pooled))

    return ConvLSTMClassifier()


def train_one(dataset_dir, out_root, args):
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from PIL import Image
        from torchvision.transforms import functional as TF
        from torchvision.transforms import InterpolationMode, RandomResizedCrop
    except ImportError as exc:
        raise SystemExit("Install torch, torchvision, and Pillow") from exc

    rows = read_sequence_rows(dataset_dir)
    if not rows:
        print(f"[skip] no images in {dataset_dir}")
        return None
    classes = shared.canonical_classes(rows)
    label_to_idx = {label: index for index, label in enumerate(classes)}
    hflip = not ({"left", "right"} & set(classes))
    train_rows, val_rows, test_rows, clip_stats = shared.pipeline.split_by_clip(
        rows, args.val_frac, args.test_day, args.seed)
    split_rows = {"train": train_rows, "val": val_rows, "test": test_rows}
    split_sequences = {
        name: make_sequences(value, args.sequence_length, args.sequence_stride)
        for name, value in split_rows.items()
    }
    print(f"\n{'=' * 70}\n{dataset_dir.name}: classes={classes}, hflip={hflip}")
    for name in ("train", "val", "test"):
        print(f"  {name}: {len(split_rows[name])} frames, "
              f"{len(split_sequences[name])} sequences | "
              f"{dict(Counter(item['label'] for item in split_sequences[name]))}")
        if not split_sequences[name]:
            print(f"[skip] empty {name} split")
            return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    def transform_images(images, training, kind):
        if training and kind == "strong":
            top, left, crop_h, crop_w = RandomResizedCrop.get_params(
                images[0], scale=(0.85, 1.0), ratio=(0.95, 1.05))
            images = [TF.resized_crop(
                image, top, left, crop_h, crop_w,
                (args.img_size, args.img_size), InterpolationMode.BILINEAR)
                for image in images]
        else:
            images = [TF.resize(
                image, (args.img_size, args.img_size), InterpolationMode.BILINEAR)
                for image in images]
        if training and hflip and torch.rand(()) < 0.5:
            images = [TF.hflip(image) for image in images]
        if training:
            amount = 0.15 if kind == "strong" else 0.10
            brightness = 1.0 + float(torch.empty(1).uniform_(-amount, amount))
            contrast = 1.0 + float(torch.empty(1).uniform_(-amount, amount))
            saturation = (1.0 + float(torch.empty(1).uniform_(-amount, amount))
                          if kind == "strong" else 1.0)
            images = [TF.adjust_saturation(
                TF.adjust_contrast(TF.adjust_brightness(image, brightness), contrast),
                saturation) for image in images]
        return torch.stack([
            TF.normalize(TF.to_tensor(image), mean, std) for image in images])

    class Sequences(torch.utils.data.Dataset):
        def __init__(self, source, training=False, augmentation="none", weights=None):
            self.source = source
            self.training = training
            self.augmentation = augmentation
            self.weights = (np.ones(len(source), np.float32) if weights is None
                            else np.asarray(weights, np.float32))

        def __len__(self):
            return len(self.source)

        def __getitem__(self, index):
            item = self.source[index]
            images = []
            for path in item["paths"]:
                with Image.open(path) as image:
                    images.append(image.convert("RGB").copy())
            tensor = transform_images(images, self.training, self.augmentation)
            return tensor, label_to_idx[item["label"]], float(self.weights[index])

    def loader(source, training=False, augmentation="none", weights=None):
        return torch.utils.data.DataLoader(
            Sequences(source, training, augmentation, weights),
            batch_size=args.batch, shuffle=training, num_workers=0,
            pin_memory=torch.cuda.is_available(),
            generator=torch.Generator().manual_seed(args.seed))

    val_loader = loader(split_sequences["val"])
    test_loader = loader(split_sequences["test"])

    def evaluate(model, data_loader):
        model.eval()
        loss_sum = 0.0
        total = 0
        truth, predicted = [], []
        with torch.no_grad():
            for sequences, labels, _weights in data_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                logits = model(sequences)
                loss_sum += float(F.cross_entropy(
                    logits, labels, reduction="sum").item())
                total += labels.numel()
                truth.extend(labels.cpu().tolist())
                predicted.extend(logits.argmax(1).cpu().tolist())
        accuracy_report = shared.report_from_predictions(truth, predicted, classes)
        metrics = {"loss": loss_sum / max(1, total), **{
            key: accuracy_report[key]
            for key in ("accuracy", "macro_f1", "balanced_accuracy")}}
        return metrics, accuracy_report

    out_dir = Path(out_root) / dataset_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    trial_names = [name.strip() for name in args.trials.split(",") if name.strip()]
    unknown = [name for name in trial_names if name not in TRIALS]
    if unknown or not trial_names:
        raise SystemExit(f"Unknown/empty --trials {unknown}; choose from {sorted(TRIALS)}")
    reports = {}
    training_summaries = {}

    for trial_name in trial_names:
        cfg = dict(TRIALS[trial_name])
        print(f"\nTRIAL {trial_name}: {json.dumps(cfg)}")
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        model = build_model(torch, nn, len(classes), cfg).to(device)
        weights, weight_info = shared.pipeline.training_weights(
            split_sequences["train"], label_to_idx, cfg["weighting"])
        train_loader = loader(split_sequences["train"], True,
                              cfg["augmentation"], weights)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.3, patience=3, min_lr=1e-7)
        trial_dir = out_dir / "trials" / trial_name
        trial_dir.mkdir(parents=True, exist_ok=True)
        model_path = trial_dir / "model.pt"
        history = defaultdict(list)
        best_loss = float("inf")
        bad_epochs = 0

        for epoch in range(args.epochs):
            model.train()
            loss_sum = 0.0
            correct = seen = 0
            for sequences, labels, sample_weights in train_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                sample_weights = sample_weights.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(sequences)
                loss = (F.cross_entropy(logits, labels, reduction="none") *
                        sample_weights).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                loss_sum += float(loss.item()) * labels.numel()
                correct += int((logits.argmax(1) == labels).sum().item())
                seen += labels.numel()
            val_metrics, _ = evaluate(model, val_loader)
            scheduler.step(val_metrics["loss"])
            history["loss"].append(loss_sum / max(1, seen))
            history["accuracy"].append(correct / max(1, seen))
            history["val_loss"].append(val_metrics["loss"])
            history["val_accuracy"].append(val_metrics["accuracy"])
            history["learning_rate"].append(optimizer.param_groups[0]["lr"])
            print(f"epoch {epoch + 1}/{args.epochs} "
                  f"loss={history['loss'][-1]:.4f} "
                  f"val_loss={val_metrics['loss']:.4f} "
                  f"val_f1={val_metrics['macro_f1']:.4f}")
            if val_metrics["loss"] < best_loss:
                best_loss, bad_epochs = val_metrics["loss"], 0
                torch.save({
                    "state_dict": model.state_dict(),
                    "model_type": "convlstm",
                    "classes": classes,
                    "img_size": args.img_size,
                    "sequence_length": args.sequence_length,
                    "normalization": {"mean": mean, "std": std},
                    "architecture": cfg,
                }, model_path)
            else:
                bad_epochs += 1
                if bad_epochs >= cfg["patience"]:
                    print("early stopping: no validation-loss improvement")
                    break

        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        val_metrics, val_report = evaluate(model, val_loader)
        test_metrics, test_report = evaluate(model, test_loader)
        report = {
            "dataset": dataset_dir.name, "trial": trial_name, "strategy": cfg,
            "classes": classes, "n_frames": len(rows),
            "model_type": "convlstm", "img_size": args.img_size,
            "sequence_length": args.sequence_length,
            "sequence_stride": args.sequence_stride,
            "training_backend": "pytorch", "test_day": args.test_day,
            "split": {
                "train_frames": len(train_rows), "val_frames": len(val_rows),
                "test_frames": len(test_rows),
                "train_sequences": len(split_sequences["train"]),
                "val_sequences": len(split_sequences["val"]),
                "test_sequences": len(split_sequences["test"]), **clip_stats},
            "weighting": weight_info,
            "validation_metrics": val_metrics, "validation": val_report,
            "test_metrics": test_metrics, "test": test_report,
            "confusion_matrix": test_report["confusion_matrix"],
            "per_class": test_report["per_class"],
        }
        training_summary = {
            "dataset": dataset_dir.name, "trial": trial_name,
            "model_path": str(model_path),
            "epochs_completed": len(history["loss"]),
            "best_validation_loss": best_loss,
            "history": dict(history),
            "validation_metrics": val_metrics, "test_metrics": test_metrics,
            "validation_accuracy_report": val_report,
            "test_accuracy_report": test_report,
        }
        checkpoint["training_summary"] = training_summary
        torch.save(checkpoint, model_path)
        (trial_dir / "report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        (trial_dir / "history.json").write_text(
            json.dumps(dict(history), indent=2), encoding="utf-8")
        (trial_dir / "training_summary.json").write_text(
            json.dumps(training_summary, indent=2, ensure_ascii=False),
            encoding="utf-8")
        reports[trial_name] = report
        training_summaries[trial_name] = training_summary
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    selected_name = max(trial_names, key=lambda name: (
        reports[name]["validation_metrics"]["macro_f1"],
        reports[name]["validation_metrics"]["accuracy"],
        -reports[name]["validation_metrics"]["loss"]))
    selected = reports[selected_name]
    shutil.copy2(out_dir / "trials" / selected_name / "model.pt", out_dir / "model.pt")
    selected["selected_by"] = "validation_macro_f1"
    selected["selected_trial"] = selected_name
    selected["trials"] = {}
    comparison = []
    for name, report in reports.items():
        is_selected = name == selected_name
        selected["trials"][name] = {
            "selected": is_selected,
            "model_path": str(out_dir / "trials" / name / "model.pt"),
            "training_summary_path": str(
                out_dir / "trials" / name / "training_summary.json"),
            "validation_metrics": report["validation_metrics"],
            "test_metrics": report["test_metrics"],
            "validation_accuracy_report": report["validation"],
            "test_accuracy_report": report["test"],
        }
        row = {"trial": name, "selected": is_selected}
        row.update({f"val_{key}": value
                    for key, value in report["validation_metrics"].items()})
        row.update({f"test_{key}": value
                    for key, value in report["test_metrics"].items()})
        row.update({f"param_{key}": value for key, value in report["strategy"].items()})
        comparison.append(row)
        report["selected"] = is_selected
        report["selected_by"] = "validation_macro_f1"
        training_summaries[name]["selected"] = is_selected
        training_summaries[name]["selected_by"] = "validation_macro_f1"
        trial_dir = out_dir / "trials" / name
        (trial_dir / "report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        (trial_dir / "training_summary.json").write_text(
            json.dumps(training_summaries[name], indent=2, ensure_ascii=False),
            encoding="utf-8")

    fields = list(dict.fromkeys(key for row in comparison for key in row))
    with (out_dir / "comparison.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(comparison)
    (out_dir / "report.json").write_text(
        json.dumps(selected, indent=2, ensure_ascii=False), encoding="utf-8")
    shutil.copy2(out_dir / "trials" / selected_name / "history.json",
                 out_dir / "history.json")
    print(f"selected {selected_name} by validation macro-F1 -> {out_dir / 'model.pt'}")
    return selected


def write_accuracy_report(out_root, summary):
    """Persist and print validation/test accuracy for every trained model."""
    out_root = Path(out_root)
    rows = []
    for dataset, dataset_summary in summary.items():
        for trial, trial_summary in dataset_summary["trials"].items():
            val = trial_summary["validation_metrics"]
            test = trial_summary["test_metrics"]
            rows.append({
                "dataset": dataset, "trial": trial,
                "selected": trial_summary["selected"],
                "model_path": trial_summary["model_path"],
                "validation_accuracy": val["accuracy"],
                "validation_macro_f1": val["macro_f1"],
                "validation_balanced_accuracy": val["balanced_accuracy"],
                "test_accuracy": test["accuracy"],
                "test_macro_f1": test["macro_f1"],
                "test_balanced_accuracy": test["balanced_accuracy"],
            })
    (out_root / "accuracy_report.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    if rows:
        with (out_root / "accuracy_report.csv").open(
                "w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"\n{'=' * 70}\nPER-MODEL ACCURACY REPORT\n{'=' * 70}")
    if not rows:
        print("No models were trained.")
        return
    print(f"{'dataset':<28} {'trial':<18} {'sel':>3} "
          f"{'val acc':>9} {'test acc':>9} {'test F1':>9}")
    for row in rows:
        print(f"{row['dataset']:<28} {row['trial']:<18} "
              f"{('*' if row['selected'] else ''):>3} "
              f"{row['validation_accuracy']:>8.2%} "
              f"{row['test_accuracy']:>8.2%} "
              f"{row['test_macro_f1']:>8.2%}")
    print(f"Reports saved to {out_root / 'accuracy_report.json'} and "
          f"{out_root / 'accuracy_report.csv'}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=CONFIG["input"])
    parser.add_argument("--no-build", action="store_true", default=CONFIG["no_build"])
    parser.add_argument("--rebuild", action="store_true", default=CONFIG["rebuild"])
    parser.add_argument("--stride", type=int, default=CONFIG["stride"])
    parser.add_argument("--jpeg-quality", type=int, default=CONFIG["jpeg_quality"])
    parser.add_argument("--day22-clear-dir", default=CONFIG["day22_clear_dir"])
    parser.add_argument("--dino-threshold", type=float, default=CONFIG["dino_threshold"])
    parser.add_argument("--dino-max-per-clip", type=int,
                        default=CONFIG["dino_max_per_clip"])
    parser.add_argument("--limit", type=int, default=CONFIG["limit"])
    parser.add_argument("--datasets-root", default=CONFIG["datasets_root"])
    parser.add_argument("--datasets", default=CONFIG["datasets"])
    parser.add_argument("--out-root", default=CONFIG["out_root"])
    parser.add_argument("--img-size", type=int, default=CONFIG["img_size"])
    parser.add_argument("--sequence-length", type=int,
                        default=CONFIG["sequence_length"])
    parser.add_argument("--sequence-stride", type=int,
                        default=CONFIG["sequence_stride"])
    parser.add_argument("--epochs", type=int, default=CONFIG["epochs"])
    parser.add_argument("--batch", type=int, default=CONFIG["batch"])
    parser.add_argument("--val-frac", type=float, default=CONFIG["val_frac"])
    parser.add_argument("--test-day", type=int, default=CONFIG["test_day"])
    parser.add_argument("--seed", type=int, default=CONFIG["seed"])
    parser.add_argument("--trials", default=CONFIG["trials"], help=",".join(TRIALS))
    notebook = any("ipykernel" in module or "colab" in module for module in sys.modules)
    args, _ = parser.parse_known_args([] if notebook else None)
    if args.stride < 1 or not 1 <= args.jpeg_quality <= 100:
        parser.error("require stride >= 1 and jpeg-quality between 1 and 100")
    if args.img_size < 32 or args.sequence_length < 1 or args.sequence_stride < 1:
        parser.error("require img-size >= 32, sequence-length >= 1, sequence-stride >= 1")
    if not 0 < args.val_frac < 1 or args.epochs < 1 or args.batch < 1:
        parser.error("require 0 < val-frac < 1 and epochs/batch >= 1")
    return args


def main():
    args = parse_args()
    root = Path(args.datasets_root)
    clear = Path(args.day22_clear_dir) if args.day22_clear_dir else None
    expected_clear = (sum(1 for path in clear.glob("*.jpg")
                          if "sem_obstaculos" in path.name.lower())
                      if clear and clear.is_dir() else 0)
    outputs = [root / name for name in shared.pipeline.SCHEMES.values()]
    if args.no_build:
        print("[build] skipped (--no-build)")
    elif all(shared.dataset_is_current(path, expected_clear, args) for path in outputs) \
            and not args.rebuild:
        print(f"[build] skipped: datasets already exist under {root}")
    else:
        shared.pipeline.build_dino_datasets(args)
        gc.collect()

    if args.datasets.strip():
        directories = [root / name.strip() for name in args.datasets.split(",")
                       if name.strip()]
    else:
        directories = sorted(
            path for path in root.iterdir() if (path / "labels.csv").is_file())
    directories = [path for path in directories if (path / "labels.csv").is_file()]
    if not directories:
        raise SystemExit(f"No datasets with labels.csv under {root}")

    summary = {}
    for directory in directories:
        report = train_one(directory, args.out_root, args)
        if report:
            summary[directory.name] = {
                "selected_trial": report["selected_trial"],
                "selected_by": report["selected_by"],
                "validation_metrics": report["validation_metrics"],
                "test_metrics": report["test_metrics"],
                "trials": report["trials"], "per_class": report["per_class"],
            }
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nALL DONE: {list(summary)}")
    write_accuracy_report(out_root, summary)
    return 0


if __name__ == "__main__":
    return_code = main()
    if not any("ipykernel" in module or "colab" in module for module in sys.modules):
        raise SystemExit(return_code)
