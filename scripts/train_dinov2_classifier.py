# -*- coding: utf-8 -*-
"""Build the SenseCV DINO-selected datasets and train DINOv2 classifiers.

This trainer bundles the required labelled-MP4/day-22-clear dataset builder,
taxonomy, clip-grouped split, and weighting logic. A compatible
``train_mobilenet_dino.py`` beside it overrides the bundled copy, but is not
required on Kaggle. The fixed test day is reported but never used to select the
promoted ``model.pt``.

Example:
  python train_dinov2_classifier.py --no-build \
      --datasets-root /kaggle/input/my-datasets \
      --out-root /kaggle/working/models_dinov2 --dino-model vits14
"""
import argparse
import base64
import csv
import gc
import hashlib
import importlib.util
import json
import shutil
import sys
import types
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

COMPANION_FILENAME = 'train_mobilenet_dino.py'
REQUIRED_BUILDER_CONTRACT = 'sensecv-deviation-mp4-day22-clear-v1'
BUNDLED_PIPELINE_SHA256 = '2a395cbd63ab2d33d82276e58b9fc8a3a4731507d42c610891cf9304b6721e7f'
BUNDLED_PIPELINE_B85 = (
    'c-qBX{c_{TasQuBF@cjZut$)RR(n3{!mHA0wP)G;m6g1%6vM(oBq-rT0yF?ht5cN9L*xnbB>8ps%wX_A-JWw!wroiNW_o&h`g?lj'
    'ext%?p_=E5G`l)d+iDqpLvIG2=Y79TR}1y>?CrbTgRo3i$-'
    'GJyY7tj)nN+1(7WrD8Cs~=Cd{m3%HjS$^&(wTdl|xlsCz+~>IL%Z%kiPNdDp7Cp%XF2zO{$Ok<Btd9kI%;+57lKnzq!n_MBSy;wb'
    '~R3cf?p{>-dTqtGgt<y2hA!Hu#iZmQ_4oCDF1-'
    'l1*G)N5DP*q~dI$K3P1XRguYu(O__%td`+C&nf}C%nS9W`08qvj8F1ae0fCZsx;edE0vb~u-'
    '&Zkc#$jypDyx6UPkl0sFE^@w^bfhMY<l_fj$A;YLOI5!v`G|%>{6i)hbzl7HGf5uy5WU4mQ;An&>LMM(Q=SE75zM2uVMIf>D|+^U'
    ';qbzz>BggjA8|mHJfX+hU%Sqj`DzX%J%_;?;^MI8tX>l@uAswceJMx=hr&w>(!_LMBj)qH2-fWi-'
    '<gRQmqUXRlu#4U~GO*72sqnwGa|u3n$Mx)^^r`{554D*R3v>Z`BRDp^+Jg1`*bCNAPdtlqwRd)jOTa7tw$$dJchrME?5o6!nJw1j'
    '|0ZALu!A^|%lrAp%YHL;VrO&3Yd-IMiYvM5!`>=&=mh>4>o_%Xc*&&T_R%CGWbem%P0Uh?qEc%EG5`Aw0ONsJF;;)~k@TA4U2L7{'
    'b0l(v=r8kTDi)>1d1@fq3;xUN*OTBNt=mZ)-'
    'j33*We{h$AZmy8uL(M_@ekx2>&*?C@q^HiCx(^Xug)xD~6T2cnMim_VUUV1Nz90PzW@;eoVgiTmo1NUoijPmzJzyDfY-'
    'h)cZc)O~Cp_<2=4cIHL7@F>}jz5QBws5{mH_AVF;dT)+-vN(AxV^lN3qY_kE_mpPrCF6T<Q<YX{)E-'
    '*5j?!#W+WKENb8J{IC#jwcyW30SWuXqd5Lw~S^`Gu0w`XxG6G@Co0O-'
    '^w7tx$YX+BZD@ZSvWmP2c;$9`6QxMh9MJ$V)BvFP7Lr_Pcjqy+<{I{k_yjY|(2`u`(Dz+Jz5Q}`TK0PawbtLoOuJZEP=*P{~Cy`g'
    '0-'
    'u=M=C=KQU@e4)>E0%AsL~cu>rd<rptU~O((7MuH4YLfg01S@g+z87!V_gX?l8~}2lj4??4zLIh4)7+wO|m$fYjskS7b1S~;rQ(Bk'
    'Z>_{7I`l>D`L}U&%h;w<g;gxA)um_O+@cLft0K5It!R1M7fV|<JA~=Lj#n#7D*8~8k*4LDQ84QV1ng^L8$I>Gaz(822ja*lNa#{8'
    'YRAgei|tPsU}uJmg#4<JojO~4A)rDYvqR_H6q#%4P&c=gW%g5F#(_oK@YF1beSd#S!rI}lJG*qA_3JF+j&YRY6(dVt1T;X?v-'
    '4nP`Av5W}xy80zf(&dQZ=ohRUH0Q!o!K8Wu4nr3dz-1;W_32}^o-'
    'xyt7^N(K<6NHqFtj*n${8Lw#RMAH<>)t1!7Pnf*4AE{%5(>)7=VHK<!W}Gg>2Cw6Jk%zDLu>z}mnXj}I25Z{e5ELb&48swUMk4_J'
    '?fy`ahP-_~#H`fqVK7psBnl#J8y+S#;AFDlUAmx^iC3HJc(Om*3u<$-'
    'hDKjOMPZ2!KjZ)3DoAc3qQdRRpd;9e;bJCR$O1?U68xxTA$J^uP+KovfGP#{q*7B^h~zVPYo1mLWWo?YG0vcQF~JH#P5PWxmS{jq'
    '7#3gwR@DxM(8~BGY76y#0S_lHUO^d;{{&_!dBw^YtI2a3-SA7MBwHsq4Qd*KeETy-XT!_|gI&_Fei7fPz3|}03-'
    'LFGtnEWc#kOPziR1$`HGb7?n%pG?YbT}PDgh%99sb1l{rv?Qx!;yV(#>ael4rN!Iom8&3YA5))R=C_Cz!_oAq=||+($4UAQ*EFjD'
    'C*x8T<^=Be1)U$95>x;Wa2(im`5lj}K}8iF8<8thXByYEUbD{lSV-0xLygm={{bU9EC(I-'
    '_w(qK*|0p;loiuao)BCWnSmK1Qja%vHWr@q7+k;9rqm#+-'
    '<h1ZT<8mTdC*bs%<yCarZTX<Uf8BN(izNF<lrQse}_**uXo#X7;s(=345uXCf2YXG$yOn|{Y|E$&McVm|u{Ez$L$L9eghjD5*!b1'
    '2TNun+Y9nb_S#jrc!pk35O8ee5_9N-JU3BqUbrC66@5f?Q6{5sFUq=cnLd7)Q(FgV^`t%+tVQ&`gENL@oi{X~8$92-'
    '*1fK4uK`10)h`1{wVFL~UG6`k?tHSPnS6%;5^yM2M-'
    'CvtoU++DtfSuOA44Fh4*^DP`8=&bm@)b8lOxDjJ((SU5zAKySxCSqq|HHRN9D+HSR2`scadH43!*$?6)1+1ay8wsZ>=S8}KSp%^)'
    '_tkZ-!;r`-Id)D)n|t+dOi%Kd#PuKF2oC%-'
    'Ha^QZE#tp#lLW?%+<tirY>I@LZ~Sxdfiq0Q4YI@C<;4y7*$(J*fLGJsmY2BXnK;e6dK`X67_xYj>Tr*Ox<N{|me}bEwhO92Kf^yT'
    'f0EN>{w{2T@*2u{WuNZL0o(XFxmI)09s1izzNJ9Kn&m}0ucR@p^(wtI4d3H&U?67OHP!@6m~94wm&X^!=cgADha{&TBCI!Dmy-'
    '*abgv}UIKK@og~Rpcu;GK<?t6m>AWowXr|3~6BQg{qjbC{FzbzgP9)r;RefSu7!C)`|zT{*?`O+to9A$84f+Hp`DEHEJKbUq4adT'
    'qr<dB!QN^h+4hBp@OeH|X`9ef@B`PE4@8jWZH2;3TI84O1fl+;5KJ9|J#T!3m@WF!gxdUOSsa^vp@@W7S;0S-'
    'VY1v@?%{07VXtNjDCx6l?83$d<Due8*CC0Y-'
    'XO(F^Xb@KWQf*)TYFbBsKWG?WvL>%|)BBaHAq8^wi1Y8FMz6rDIK#{#lLXs?$h=>ub03BsHS;oC1<^6qx-'
    'V<ZwWqHJSVx(6wmTM@YiWyKCBi<9w1A00w5*Rn%s)0r*Bi!KWXr&Df@Ss@dq=$72id!;b6F0hY0SY|$k{}tt?ORY9o&@1K{CNwGm'
    'zeM}!9WSeFxsWWYz5Yjo8&6`>oz7|3=n_+LeqdHGh)faY81ba=--'
    'UJf!g)Y6F3pv0*O5$11o}4G;YMC95tDBA+CsQPz{tQs&>C2e2(BhaNvW_2WB!WMDWBnZ#kpk5==XJ>~gEOAP%OIk#FHIuq0PO&AE'
    '~mGn{BKCPnWOr)?L!Ap;<U#VyevC`=3`n^ZPI614u9eb^1x4QXI=cB0~#9n-'
    'uvtJ>yJKPccK;d;BT{+(<Pemtiihg$g7S5b+mh6sCbr~z>lK%6+*({IFX;nkseBMxR{KYXL#U&ht^8qd!U^b2@wm~cGjmEC{g$_d'
    '`GS%z0hL_{ho6t55~U8?H@vI7#ZBXp}>)ZcLImwd*;sb(GkmJ9(7Wc3mX5DNiM`rZWMeet>mXPqOhxJuKNVq)ta)s7a%j@O%2Qc4'
    '`)5yD{gdnTZsa+xH=$cN8|dJq$qL2cT&v`FUZCi#{F(MST(3H<3Lg-'
    'M(Xqc9zUO0nj*8V=hf6PKcSLNcz{8yvIYL#AfMybc2ag_37i9PDFyA%TX6S^J@80!0e_MXb_*4yJk>kTrV=%r6s)bIIT5poT;4Be'
    '5^+D=*R|&;?)-sAjl6<QOIf8pTxtfSfCUOoWRfCCzw!59y|)<vM|@qQnFvx?U{Eg)|X1!pDR?Pd3Xb<-'
    '!0<tkKj=V#awSc_5cO&E~7^f|`$zZ#g@DeQphxXVwHWf3RxJqYyA>bF?=)*c)0tUy-uhPeR1t9)(gAN!gcLZWzEeBWC*vAF-'
    '|khC0X_7L5!kPHQvjZWP9^A&0D!k-'
    'lAfG&V=V5%fRv*Pana5kUS3Qd5sK;{+&a7S?t5)ZJZdYJ?f5bI$=m0Xs2EExwONw_2v$O`hj3TIBg1QKC(r=SleD3GzJOL7$6a`y'
    '_epdg!y?9O{=7NF-(lg?`thP*YfT6KU^Di1gxVA|1Z?^@w!v1d(`7tWAC~b!z^B);K@;!|9t-mbI?Y+ieLQ&`d_xE>f=^L-'
    '{y(_x<_B@yYAc=;ZbB`8f|d^{l?9d6>bx+2G~r$Ft*$vv+SBtx364L)P!~2fHV;!JBvApS?bPi`eJm>4$UlqUOF>RPXkHe?I4*hX'
    '@gz)`!u>*_$`fn|CiyUsIT54fE|G9M$YLVl_HS4i5MBq5^go8TS1fPlr7m&qN(#?sF=xj)9KnbK0i>ym<;We7ipy-'
    '5$!$jLEvy<Km-(1nbb!MnJ?(a20=IlY&*&BaO%--'
    'T1)>fqhcoH!>nCY|QvxNR#%Q7m>VRXmE6EfPfJrkwYmtzb1pHXVDqFewd<Rv&XCpoicRYMXF)zh^_7*aa^3T*^;1pbX;6*iO(-'
    '!hE9tjFZ`OpbxOz0z=C?TmEn+9A((WY59<I{9R3LYi%&5?6;&7m`EU67^*X-M8Ja~tXFM=1lX0x#&v}-'
    'wd54O&R}|$fS4ntHskn;FX;rTxk3bAs52d5m_Cj)Vxc?F{P#40e-'
    '56ewZ49bw@Dmb4+KcpRGUKZQ4J~M!()sQzMJ<eA7?!);P<Ws73&4oj4xVbl0^Oa7?38*(e@kE9J?-'
    'h*P(5{D{aIoaNfa=Pl&kGky1e(D{Dy(#jebP()t}S2{(OXpY=y*(PpM2bC)W4E$O|O(010F_ncp)k4WGiyNN-a|aB<-'
    'e`V>;s{P#hIlU=(=Z=h4&m>XlG`%WOcJK=UK*2J`i33&r~BD>>bn2dUV+^fa6;%1mow4_8(&>`Z(BfpUn1ZtXZpbcPYxHMwoacM+'
    'YrZ7gx!QPP<Hv#MN&RiE*F@l`<USon<foWbpQU%YkAnMWX+)Jd%K7eu{)-rh1q=d;{L?%>-'
    'C{$=s@?~1iDfyzS3g#TTa4e_`2OM$`t~T_j>jn%RHSs6A*N>ZVyaE#@x{;=!*AMe3IeYDI_%%fayecE4;|s`{&xxxIEdoC|i=Z<%'
    '69Yk4N6<cj?>!qK!{-HoE4H+v(y1w|wB4I#+oWy_{sCNOfA?La*+wG|#%XX{Lr2*5ef-'
    '0FDaHL!CrzP8<~?k#vG8j+OyUY5Wt+p?Y07RETijFrMOq4Lh<8U6nPvH3<0FN<{odYwXY@@$NyDW#g(H~FuoX{3@)pyBMdJ}JB!E'
    '5Ars+KgP>$@>crd?ch9K8sH=$@%=u49rnh?TCEEsxuMQONA+e3AP`N@G=b4KP*SRFxLOi2-D%LixhM=yAqdx+U-'
    'gDlAw{)4CG_((Gid11tYcp<Df-'
    'J1=`?O)O>`Zes&U3u;_(bRo>Y$SeZe3O~xj9es%E`QFx0gcKl$s61a6aP{YcvGJ{V7(T}=OJw!CJ|J__dk4)j9xG^No-'
    'T5gdwd;Kxqk90b`9m(oiSy23RFN`)hXWAaK<zx<E?afnpHju9}^KLWn6vcWg-FmynIhr*6a<G}g)S`{@0LckiQD@6XxtphKX`3b^'
    'RUY9F2z9z)@OsfnNnr5-oG8S32SgmEJ=@H?smscuqarh138MU%BI(eK7lAT;h*6SeOGk|j{zseL7-'
    '4%FzM^y{BdQ!+EdYQA1B0WoGuVF3E|o!S%gOOd%=!@w*SW{31mnfMJR64*oP?pkO!F)Mj2-'
    ')3oDQynjK8n~7F`s4c_)MmAnax55=P_NF(`A4S;rRF6A8w)`@Kh9t@xB7Fvui>(l{fu>Lys+k^;T+n}PihFAs^iAm6U;GDg1^Kr)'
    '&ly`0&;bQW#)lw=NSDmF>ZwR!nKX5aiw~ZjNOzvF>k#Tr&FJyajK=FU+SR%RFe?wdce)<aX!#C=vp&MU1cIhyn5&?J`S~~58~bgt'
    'XI3Xa7h_Tmwh}M+5k0C!cnijG~}I{<}-Hp?j!^Ox76KsSO9T<P-;N(*;sU8e5|uWSlGk%rUNN;+TntYq=dy-'
    'G(~M!NQD|gloX2zm(b3B(^um0gfD7;spb#k@SwcR3z{6=p`X}52jXZqDLeXx)_!6qz$@7TGz2f#9KWU%kZh*v{mR0Nq9wLZzY6{l'
    'OYd~X*8MtYDz?ZdYFeF|bLvkX>v^9?vHP67&B^16!gE|Bv%R7*+FlAez~&#B54#9Ci383YM2<K+Rc;iSS_GZ<o~`ES`M30XwtOE9'
    'j;`YPjXKR_s}5X<<h|Xei)oXWwy9T0NRQWP$rI<oC-'
    '#L%sisuPLwoCsQ6qwtEo%qaK>fo6z?&@4oaE1;?5H|oV(SAnKzcg2R@0iZj9MQYwOO<24K=8H<f6xk$C@>cTpnt{5Pt?u?3(E}fu'
    't7d=1sp!W;OWrch04V+7x-'
    '}LRz~~`s<E1;IyfF$@R+Tx}MY;?o+Bpd$LTLoD`~Nu8WzeW+Pl|r5?;yppHbH9sknm=BFx~rJ~qw!G7Yx=5M%)noB0(g#smUDe3_'
    '**ngni?_Pw|`h_l6(uIFZgqq;DKuR^2X;D@+#_{ToN{|>CJ}Kbt@O~oSNn+zU3E0TcP$o_uQv~))t0?lZn9HGJ_uFDWA#8@q1v$Y'
    'pr_Fh%bMpMaRpwIh!bR$J9QXe9$=>1qi-Du#R?86v5)ZujTKY}|M(dNU6`#}?Hk-JJk`-'
    '_sn+@w3pjrLUY(yz?6Br`1MJh?^@9B{onfNvaV_cFqqccbI&DLw7#08=gd59=nj`$Ulf>rK2;{>Awsbb$%skX{gCtXWU*bhyu>5x'
    '1i&HG4GQyJ{2HFy$dhM1#@3qJkn-std~7Zj`<zWADd?R`B9MpTgLBQ=YxEY)~+uxkP?t3?mogM;4_Kt_#zz4#??t%pN<5wng=ONj'
    '^a1P@N{N8^z&Cp*NYHV$KU(imD>)+|S7Dma+|>)n5<f^4h&72QH!_YWQG*PG0EI+f9yu0;artCV)n{Rc$MHh<wB=%8||W^~V9TI0'
    '-zmc(OJ@VsFL+%1jt?|O;Vw2emuFRH7OYmzjMkQmxqSB#Z*^oLh}DdLUqkT%OkfV=JWzI#;AdWW}+QUcyNvCV`2X3Y2`{e#}oKqC'
    '0i3q9E99fP?Hvi3g*qfMT#ph_h1Ip||1GxO`E%amL@tp^jE<?<!bhn23;5c-W#k+rv(#BO%dhE1Bnd3068i?(yjemN&AW7IDA2&-'
    'u$NMkW<Yr@{xNStHoGEI+CDurlvn{|4TLTIAuBD9qF2;Fy3{_0YvX6CmQ%4VTj#4e3a-'
    'o1YJAwq`d;KL8!_b)zFH2RNl?q`!sj}B(!FZ!&H@MFhPHwT5(V^Tg|44s~ou|TeOgDPH<?+LX$dwYQj&f^cK$Njhv8H{@=%fxcU^'
    '`HbFJe6Q8yGneR+&$Kfk}?VqMx&74<c8dTpRLxVaOuXUF?Op|x0nA}g|$(jA&esLGzZHXb`)*;IoQ$0%J*`c&E<U2$Zin&Xam11T'
    '~GGA{h6N)b<l7G%jpN5M1;1tAPRR7aC?q?i3_?*zToM|JES7%+!Sh~F0Tq~R~GP@MoM;TO3FrYwLH8#b{l;z(EcP{KLs|^<D_jg*'
    'gier@IB=|YKfz*uKoTldvza&`dWGD?zRQ{dXq*cc1S1G*L~ZP|NVyix1>R%O0Wa<-'
    'K6VJ^=OAi>QJKbI`QckfJI*UgyE18OlPgF0Lp5i<3G3DyzYR6oZ*?Y%To>!(pj66sPEq|Ec!*&y1#a`L~_8zu{?||74@;FU3T*34'
    'ubYbiF;O({I0O~z@!xeQ&|tI)$dl|R03~W>WmuUG2)U~N><=B)UC<mnwrc;%SPd7yNBH(iDUz=lOmOZg!6mKjhuc?$;CS3QxNsR1'
    '-&!oCPOtmlhM#yfjyg)x}imeG3MggqsHC}Srh!49@PzIJ+3ZQ(9RTpmUAzajqn~w90^A7C^ej3b2|cGo}BgDyvdb%_f*^l9NFfyU'
    'a60=wh3S`d`0wWpwbJo3&bujmWPP(KzWj*dcntfkDPUsWaab=s4HFaMBA(D(hnYOlF>Hggn63T5lquYVo`q_ab0-'
    'Xc8=uS$Ea9WC^GPM+ghoZ&g&auFiCHh6D|^yLh~rU;m5j*IY=~7ojNL7G|n+kdg`@22_8H3R1H1Oz~vYPrM6|L=8ViFK-'
    '{`@pl8s+d7xcoPOSjrX0D~K+G*tjL_ixw=B@NV-NJnx42=H9^4~cOF`e0kbyi+h+?<HEuB5x2NHFw)9IYC-PM=F(+5030W;gB4Up'
    '}1SK*}HApZ*a2%b$;5pI!Vb{5i(cE$!E-'
    '>@=Y=bgp}I^8Y+C{ZZEg(G#WJ`L~ntYj2G<Qhr05*&DVtOJ}uDlRz8zC6c7X?A&O`8w%RN8LHl$PsU8@p;{=|131G51s!=XQ|0{W'
    '1C8~Fr1OI_&e7hB#Up3$9mBek)4$w_AwP~E3@9hliW(>6=wn?UWkhx=Go&hG1F7v~(ekBhHc`Aa>i`NYF`Q*@Nw_G*58x4yNUG76'
    '-U%P&Bj{rK7z!+a1KZ7OTb4EG?noFr$zd~_Rz_D?NV^KHZw^-->4j{il;t!_cX3<E=6b0IJ9+R6)6QE8TQqMS27Qz1-'
    '{)^+rY%p;^Ub|%$+kJ0UpDsB``C(GO@C|1{M*_8iodk(C!`K!lWHm{|Jl%m_W#NTx+t!n5j?tS#QPam;Ay!pXHRm4?%=+h?P3qzR'
    'zJ_4<_>*Qs<qA6<r6jpTZB9)K$(jtkl>k*OZamA4p+b;o;+%oLE-'
    'aM3RIj(N*8f*Ne}O#I0SYQnzVQ7wqhDPo`A8}nqi&FX)eN<bV^foRqCLlfDNL$-e%U^Qzn<CF;;)YhBk|J^-f3j={#Lra-sPW_3t'
    '+sF2hmv+$RtG_oMe1^rs{rWEGC~4i^wT>-_QaOiuVc*wIjHWrh&m-cfnfst6gX2-'
    'N7Q{b2%0`O!?9X<Y*JB{BFfB92Ev8DYTXV*>71BT8oO4f(lJEKWI0tTQ&!DUGSbUL+LJkp8E)d`#KPtOiR4n;_`&*ZPca!-'
    'F(Io!+xfGak8>G3@g!OqG$&0yg0^lYH#D^mw?#juh>5Xu7$yJ`cxTQ0ADXGZpk;)jR^xk>me6lp1dAw5RE;S%LM1Ea-'
    '5bnraarKM-p8*sl9ndPq+^NX94G1R9)94LzFORWf^ai^Zv`vUQBR>aBDj-#8!}IMq^@_hygIIZN-'
    'a?15m3c({%w>lxXJwOw%2KAUGvwKy8>xzy1`T}f)c>^aEPYfFD}6w@wr(P)@pG&pKn0O~WM4xst{WEHP3akCg+@+N=F@V4qqTKRO'
    'oxWhzJ(^)G})MT?`_K3JncBgBu8YJJk;RSUQbWc<e_R_Awz&ll6w?I(p{u&MH$@Z+Ii(YG@&LSfPP*KMk)Krw0q)3fEY^z|CDIGk'
    '?lg=vFPHwLbY@BCkO~UQl)>Lxzq0vRkV|I!~Q^oW<V60_G>K#Dc?bv2M@S)vaD~>KMP$FHO(|^dAw=YGVxncwejf%D|I62zv*SGD'
    'B`L?}p)kB*f9~}-2yhi#%pW2a`x;xIgI0N?{;qD*5&VfAiBtagBs^>WW(nAhJJ@ltU9>-D=q%R~8A4db12fn?-'
    'T5laSeNpmDYlcqL(%d_A@<nddW+3&K9vygW7x6!JM1<i|Oih~8)L#CDU(gO7XI8k3a3V>Tpr_}Ee_cz#=I#bGC#q&EUHL(r?<R#<'
    'r;0~9bE`sI-JZ<Z3EBR}teuWbS`i{A>WKyD`aT@lha0-<QkYg$rSszGptEFVRog*YT+%_~S_Or-Wk^_ug7(kNtACSJ>Ral&Y3RIJ'
    '$fw0Z)=upVfhH`w-'
    'uzb^OVl*2w7WpzFgT7^rVmUci%SQCC2J_p9Ic5>#+7Vt8c`4VxH~{LV?HpOG#rLDvpfd<VCwZC`)M`?o(P3$O!0HvmqBNl+p4rlr'
    '5;uL6a?$6?||_I!@KlqYUn5WIvUsx-PX!abZ2#G6?EfBc+Eo2JtuvM>28nK?I-<;x!LcrH)rBvX8(3F8iSskyw|w_6snfhnH-pEj'
    'Khbk7AO^v!&P!(BC93`q?1dWRf*zqo~9GN`XJ~k>aHPI>Y=4K=#-R|yGLV<)q^uOv#Us35ZHYk15z$hUXVxF-'
    'L4wSD;<#Itx4xo;9>TG9_vjl{rRX#I(zt0zRDeEUrsyS?QbC8ufJ8yca$FSF0i6XGpg(QifI*cVK|{#-'
    '{vVdiT*vdiaBP}DJz(1b1_r<bc~b}lOr*GkP2v<u!ZwtPgm*QxUSK;kkuZIDR0eD5oeUXQzFe+hTRj>x%W6o$NSiQ<g|5Arh`l2Z'
    '1v3AaStD`-E}&0XH4S;D-'
    'WiaoU8;?W}N~qtba0m;wmVutBU^0eSK!dokGCp0$OF^K*$`DCcZKnoMnHg%(X&8Wp3c1>M%&H<%e_X#kZl+<!JBEm0xGz=n7T7(w'
    'ZWANH;F^nbVb|bUP4_Hstdv`VL%CA<yJ0Tq@V#BUs$`?wT$LsBhkD-i!-x#bAdDowdF$#N3YRc-'
    '`c+(+xL6zS*YKd+K!4F5QAemk$zPefQ|Io_lrI`q~|>$9N8|X!2#kRP2b$TMhbnY~exc9~Yb@;QI!ALuD#Mk~_eu*Kg@s(#*Z92_'
    '!_*pf1+tg#<ph(*Y;p_+r<Q&YMJd@AFbQw;J(@b50=p0+Fs2($^ixdm;~tOXC}<{aD98=3L+YBVDjgLIn9R`5I&qOrl#K0;@A1Pv'
    'vh4<3RPKL8jo?R(hrg8p#%1tXej*7~LqRMr}%e&Rple<@z%!VAORE2Eu6+a1_qrpjL{?rR3IFVlAN-'
    '<~gq~0B{IUpGlbqE;~L72$m^gi>I?%G&jI&!tO0#kn0!aG*pdW>-'
    '?r3bP6y|s*ASVWnO$qsODtiKr=@uK%a|1E5F7LK!%zNiETCPe$o>Vnw#*LQ@V%ioUy^STtZ>CK4og6y~yV4NTd&57!|3>St{Cp%p'
    'jgdY67_)GQ&a91Bpibee+`99Hi=k#HA1XH<u|;2`p}?Q*_lNWCbb-'
    '`;6aB_T{_<QOZrR38`>*$AUMY=wW)qz?}4)L*HggZg@0n?18dkll(mB-gAh!Xi5&(vl>aU41=z-'
    'LE|vB2JGNM*XQfEb|%idnw#*_xoydTpdJ1uOwR(-pB>;16bsz|afXwtjkGaxZ<31iUDZ^5sr?hIR-'
    '|J&O{`9sUpJOD;kZ(>S^Jy0!~(Yr_|k3+J7z`zuL<<&d?qJ%>nL*`*t5w&r^I?*msr;5ni-lq-0bYc;z|gJ`!&-'
    'Ac3P;+wtI;PU9Knhfp7HPA$%%1mg~XiFspw&$1Q(+7aex3`F>!&imq1DqK~dpx-8qVRQj$vp@F^TPT$zyY^_huIR6n4+&f-'
    '1z|{BT;bv=nSE@9i6H73_I7&df5kOtacyRO8_!<+V{}utVYuO^kKCM0ZC|B%nE1o-F-k@)@M^sm68`-'
    'pt_v|{lB%Rg<VAd2V@$l+J7fx~e+K7Z!g>xj``_5{~b(DTDGv7Cu*i|oE9vv1vF1aAtpITjhFG$}L()T5{LD7n<YH{4Etuq7?q}U'
    '^TA#n5yl4^kAjrsGjF|&V|G%hJ?7ZZLYxTb36&w7W~-'
    'f+es5cSb78y9jI;|iz0YfXl=S__)%Q+pkyj>v_$2IRd^20Nnqjz&FdSNdycy~K77Vf1N&CIUN|Yo?xceRsiV`bD#!1>F#ay>)uiv'
    '-0g#ueGkb{DVgL;7r7%pjxk8?G2Mjg#%o~EyIk+iUy?qVM9CLU{-@hf1EGqG{QUEid5tO0IR52Jp'
)

def _load_pipeline():
    '''Import the companion from scripts/, including notebook environments.'''
    try:
        import train_mobilenet_dino
        if _is_compatible_pipeline(train_mobilenet_dino):
            return train_mobilenet_dino
        sys.modules.pop('train_mobilenet_dino', None)
    except ModuleNotFoundError as exc:
        if exc.name != 'train_mobilenet_dino':
            raise
    return _load_pipeline_from_file()


def _is_compatible_pipeline(module):
    '''Accept only the labelled-MP4 plus day-22-clear dataset builder.'''
    required_symbols = (
        'build_dino_datasets', 'find_labelled_cuts', 'decode_frames',
        'read_day22_clear_images', 'read_labels_csv', 'split_by_clip',
        'training_weights', 'SCHEMES', 'CONFIG',
    )
    return (
        getattr(module, 'DATASET_BUILDER_CONTRACT', None)
        == REQUIRED_BUILDER_CONTRACT
        and all(hasattr(module, name) for name in required_symbols)
    )


def _load_pipeline_from_file():
    filename = COMPANION_FILENAME
    candidates = []
    if '__file__' in globals():
        candidates.append(Path(__file__).resolve().parent / filename)
    candidates.extend((
        Path.cwd() / filename,
        Path.cwd() / 'scripts' / filename,
        Path('/kaggle/working') / filename,
        Path('/content') / filename,
    ))

    # Kaggle attachments live below a generated owner/dataset path.
    kaggle_input = Path('/kaggle/input')
    if kaggle_input.is_dir():
        try:
            candidates.extend(kaggle_input.rglob(filename))
        except OSError:
            pass

    try:
        return _load_first_candidate(filename, candidates)
    except RuntimeError:
        print('[setup] compatible companion not found; using bundled '
              'labelled-MP4/day-22-clear dataset builder')
        return _load_bundled_pipeline()


def _load_bundled_pipeline():
    source = zlib.decompress(base64.b85decode(BUNDLED_PIPELINE_B85))
    digest = hashlib.sha256(source).hexdigest()
    if digest != BUNDLED_PIPELINE_SHA256:
        raise RuntimeError('Bundled dataset builder failed its integrity check')
    module = types.ModuleType('train_mobilenet_dino')
    module.__file__ = '<bundled train_mobilenet_dino.py>'
    sys.modules[module.__name__] = module
    exec(compile(source, module.__file__, 'exec'), module.__dict__)
    if not _is_compatible_pipeline(module):
        raise RuntimeError('Bundled dataset builder has an incompatible contract')
    return module


def _load_first_candidate(filename, candidates):
    checked = []
    incompatible = []
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in checked:
            continue
        checked.append(candidate)
        if not candidate.is_file():
            continue
        spec = importlib.util.spec_from_file_location('train_mobilenet_dino', candidate)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        if not _is_compatible_pipeline(module):
            incompatible.append(candidate)
            sys.modules.pop(spec.name, None)
            continue
        return module

    locations = '\n  - '.join(str(path) for path in checked)
    incompatibility = (
        '\nIncompatible companion(s) found:\n  - '
        + '\n  - '.join(str(path) for path in incompatible)
        if incompatible else '')
    raise RuntimeError(
        f'Could not find a compatible {filename}. Upload the labelled-MP4/'
        'day-22-clear builder with this trainer, or run this file from the '
        'SenseCV repository root. Searched:\n  - ' + locations + incompatibility)


pipeline = _load_pipeline()


CONFIG = {
    "input": pipeline.CONFIG["input"],
    "no_build": False,
    "rebuild": False,
    "stride": pipeline.CONFIG["stride"],
    "jpeg_quality": pipeline.CONFIG["jpeg_quality"],
    "day22_clear_dir": pipeline.CONFIG["day22_clear_dir"],
    "dino_threshold": pipeline.CONFIG["dino_threshold"],
    "dino_max_per_clip": pipeline.CONFIG["dino_max_per_clip"],
    "limit": None,
    "datasets_root": "/kaggle/working/datasets",
    "datasets": "obstacle_freepath_dino,deviation_lrn_dino",
    "out_root": "/kaggle/working/models_dinov2",
    "dino_model": "vits14",
    "img_size": 224,
    "epochs": 30,
    "warmup_epochs": 5,
    "batch": 32,
    "val_frac": 0.15,
    "test_day": 22,
    "seed": 43,
    "trials": "linear_probe,last_block,last_two_blocks",
}

DINO_MODELS = {
    "vits14": {"hub": "dinov2_vits14", "hf": "facebook/dinov2-small", "dim": 384},
    "vitb14": {"hub": "dinov2_vitb14", "hf": "facebook/dinov2-base", "dim": 768},
}

TRIALS = {
    "linear_probe": {
        "dropout": 0.20, "warmup_lr": 1e-3, "finetune_lr": 3e-4,
        "unfreeze_blocks": 0, "weight_decay": 1e-4,
        "weighting": "inverse_frequency", "augmentation": "basic", "patience": 6,
    },
    "last_block": {
        "dropout": 0.30, "warmup_lr": 1e-3, "finetune_lr": 1e-5,
        "unfreeze_blocks": 1, "weight_decay": 1e-4,
        "weighting": "inverse_frequency", "augmentation": "strong", "patience": 7,
    },
    "last_two_blocks": {
        "dropout": 0.30, "warmup_lr": 1e-3, "finetune_lr": 7e-6,
        "unfreeze_blocks": 2, "weight_decay": 2e-4,
        "weighting": "clip_class", "augmentation": "strong", "patience": 7,
    },
}


def canonical_classes(rows):
    present = {row["label"] for row in rows}
    for order in (["free_path", "obstacle"], ["left", "right", "none"]):
        if present == set(order):
            return order
    return sorted(present)


def report_from_predictions(y_true, y_pred, classes):
    '''Return metrics without depending on the companion training backend.'''
    y_true = np.asarray(y_true, np.int64)
    y_pred = np.asarray(y_pred, np.int64)
    cm = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for truth, prediction in zip(y_true, y_pred):
        cm[truth, prediction] += 1
    per_class = {}
    for index, name in enumerate(classes):
        true_positive = int(cm[index, index])
        false_positive = int(cm[:, index].sum() - true_positive)
        false_negative = int(cm[index, :].sum() - true_positive)
        precision = (true_positive / (true_positive + false_positive)
                     if true_positive + false_positive else 0.0)
        recall = (true_positive / (true_positive + false_negative)
                  if true_positive + false_negative else 0.0)
        f1 = (2 * precision * recall / (precision + recall)
              if precision + recall else 0.0)
        per_class[name] = {
            'precision': round(precision, 4), 'recall': round(recall, 4),
            'f1': round(f1, 4), 'support': int(cm[index, :].sum()),
        }
    return {
        'accuracy': float((y_true == y_pred).mean()),
        'macro_f1': float(np.mean([value['f1'] for value in per_class.values()])),
        'balanced_accuracy': float(np.mean(
            [value['recall'] for value in per_class.values()])),
        'confusion_matrix': cm.tolist(), 'per_class': per_class,
    }


def load_backbone(torch, name):
    """Load official torch.hub weights, with Hugging Face as a fallback."""
    spec = DINO_MODELS[name]
    try:
        return torch.hub.load(
            "facebookresearch/dinov2", spec["hub"], pretrained=True), "torch_hub"
    except Exception as hub_error:  # noqa: BLE001
        print(f"[dino] torch.hub failed ({hub_error}); trying transformers")
        try:
            from transformers import AutoModel
            hf_model = AutoModel.from_pretrained(spec["hf"])

            class HFBackbone(torch.nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model
                    self.blocks = model.encoder.layer

                def forward(self, images):
                    return self.model(pixel_values=images).last_hidden_state[:, 0]

            return HFBackbone(hf_model), "transformers"
        except Exception as hf_error:  # noqa: BLE001
            raise RuntimeError(
                "Could not load pretrained DINOv2 from torch.hub or transformers: "
                f"hub={hub_error}; transformers={hf_error}"
            ) from hf_error


def transformer_blocks(backbone):
    blocks = getattr(backbone, "blocks", None)
    if blocks is None and hasattr(backbone, "model"):
        blocks = getattr(getattr(backbone.model, "encoder", None), "layer", None)
    if blocks is None:
        raise RuntimeError("Could not locate DINOv2 transformer blocks")
    return list(blocks)


def train_one(dataset_dir, out_root, args):
    try:
        import torch
        import torch.nn.functional as F
        from PIL import Image
        from torchvision import transforms
    except ImportError as exc:
        raise SystemExit("Install torch, torchvision, and Pillow") from exc

    rows = pipeline.read_labels_csv(dataset_dir)
    if not rows:
        print(f"[skip] no images in {dataset_dir}")
        return None
    classes = canonical_classes(rows)
    label_to_idx = {label: i for i, label in enumerate(classes)}
    hflip = not ({"left", "right"} & set(classes))
    train, val, test, clip_stats = pipeline.split_by_clip(
        rows, args.val_frac, args.test_day, args.seed)
    print(f"\n{'=' * 70}\n{dataset_dir.name}: classes={classes}, hflip={hflip}")
    for split_name, split_rows in (("train", train), ("val", val), ("test", test)):
        print(f"  {split_name}: {len(split_rows)} | "
              f"{dict(Counter(row['label'] for row in split_rows))}")
        if not split_rows:
            print(f"[skip] empty {split_name} split")
            return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean, std)
    eval_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(), normalize])

    def train_transform(kind):
        ops = ([transforms.RandomResizedCrop(
                    args.img_size, scale=(0.90, 1.0), ratio=(0.95, 1.05))]
               if kind == "strong" else
               [transforms.Resize((args.img_size, args.img_size))])
        if hflip:
            ops.append(transforms.RandomHorizontalFlip())
        jitter = 0.15 if kind == "strong" else 0.10
        ops.extend([transforms.ColorJitter(
            brightness=jitter, contrast=jitter,
            saturation=jitter if kind == "strong" else 0.0),
            transforms.ToTensor(), normalize])
        return transforms.Compose(ops)

    class Frames(torch.utils.data.Dataset):
        def __init__(self, source_rows, transform, weights=None):
            self.rows, self.transform = source_rows, transform
            self.weights = (np.ones(len(source_rows), np.float32) if weights is None
                            else np.asarray(weights, np.float32))

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            row = self.rows[index]
            with Image.open(row["path"]) as image:
                image = self.transform(image.convert("RGB"))
            return image, label_to_idx[row["label"]], float(self.weights[index])

    def loader(source_rows, transform, weights=None, shuffle=False):
        return torch.utils.data.DataLoader(
            Frames(source_rows, transform, weights), batch_size=args.batch,
            shuffle=shuffle, num_workers=0, pin_memory=torch.cuda.is_available(),
            generator=torch.Generator().manual_seed(args.seed))

    class Classifier(torch.nn.Module):
        def __init__(self, backbone, dropout):
            super().__init__()
            self.backbone = backbone
            self.dropout = torch.nn.Dropout(dropout)
            self.classifier = torch.nn.Linear(DINO_MODELS[args.dino_model]["dim"],
                                              len(classes))

        def forward(self, images):
            features = self.backbone(images)
            if isinstance(features, dict):
                features = features.get("x_norm_clstoken", features.get("pooler_output"))
            return self.classifier(self.dropout(features))

    val_loader, test_loader = loader(val, eval_transform), loader(test, eval_transform)

    def evaluate(model, data_loader):
        model.eval(); total_loss = 0.0; total = 0; truth = []; predicted = []
        with torch.no_grad():
            for images, labels, _weights in data_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                total_loss += float(F.cross_entropy(logits, labels, reduction="sum").item())
                total += labels.numel()
                truth.extend(labels.cpu().tolist())
                predicted.extend(logits.argmax(1).cpu().tolist())
        report = report_from_predictions(truth, predicted, classes)
        metrics = {"loss": total_loss / max(1, total), **{
            key: report[key] for key in ("accuracy", "macro_f1", "balanced_accuracy")}}
        return metrics, report

    out_dir = Path(out_root) / dataset_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    trial_names = [x.strip() for x in args.trials.split(",") if x.strip()]
    unknown = [x for x in trial_names if x not in TRIALS]
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
        backbone, pretrained_source = load_backbone(torch, args.dino_model)
        model = Classifier(backbone, cfg["dropout"]).to(device)
        weights, weight_info = pipeline.training_weights(
            train, label_to_idx, cfg["weighting"])
        train_loader = loader(train, train_transform(cfg["augmentation"]),
                              weights, shuffle=True)
        trial_dir = out_dir / "trials" / trial_name
        trial_dir.mkdir(parents=True, exist_ok=True)
        model_path = trial_dir / "model.pt"
        history, best_loss = defaultdict(list), float("inf")

        def configure(warmup):
            for parameter in model.parameters():
                parameter.requires_grad = False
            for parameter in model.classifier.parameters():
                parameter.requires_grad = True
            if not warmup and cfg["unfreeze_blocks"]:
                for block in transformer_blocks(model.backbone)[-cfg["unfreeze_blocks"]:]:
                    for parameter in block.parameters():
                        parameter.requires_grad = True
                norm = getattr(model.backbone, "norm", None)
                if norm is None and hasattr(model.backbone, "model"):
                    norm = getattr(model.backbone.model, "layernorm", None)
                if norm is not None:
                    for parameter in norm.parameters():
                        parameter.requires_grad = True
            return [p for p in model.parameters() if p.requires_grad]

        def run_phase(start, end, lr, warmup):
            nonlocal best_loss
            trainable = configure(warmup)
            optimizer = torch.optim.AdamW(trainable, lr=lr,
                                          weight_decay=cfg["weight_decay"])
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.3, patience=3, min_lr=1e-7)
            bad_epochs = 0
            for epoch in range(start, end):
                model.train()
                if warmup or not cfg["unfreeze_blocks"]:
                    model.backbone.eval()
                loss_sum = 0.0; correct = seen = 0
                for images, labels, sample_weights in train_loader:
                    images, labels = images.to(device), labels.to(device)
                    sample_weights = sample_weights.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(images)
                    loss = (F.cross_entropy(logits, labels, reduction="none") *
                            sample_weights).mean()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
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
                        "state_dict": model.state_dict(), "model_name": args.dino_model,
                        "hub_model": DINO_MODELS[args.dino_model]["hub"],
                        "pretrained_source": pretrained_source, "classes": classes,
                        "img_size": args.img_size,
                        "normalization": {"mean": mean, "std": std},
                        "dropout": cfg["dropout"]}, model_path)
                else:
                    bad_epochs += 1
                    if bad_epochs >= cfg["patience"]:
                        print("early stopping: no validation-loss improvement")
                        break

        run_phase(0, args.warmup_epochs, cfg["warmup_lr"], True)
        if args.epochs > args.warmup_epochs:
            run_phase(args.warmup_epochs, args.epochs, cfg["finetune_lr"], False)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        val_metrics, val_report = evaluate(model, val_loader)
        test_metrics, test_report = evaluate(model, test_loader)
        report = {
            "dataset": dataset_dir.name, "trial": trial_name, "strategy": cfg,
            "classes": classes, "n_frames": len(rows), "img_size": args.img_size,
            "dino_model": args.dino_model,
            "hub_model": DINO_MODELS[args.dino_model]["hub"],
            "pretrained_source": pretrained_source, "training_backend": "pytorch",
            "test_day": args.test_day,
            "split": {"train": len(train), "val": len(val), "test": len(test),
                      **clip_stats},
            "weighting": weight_info, "validation_metrics": val_metrics,
            "validation": val_report, "test_metrics": test_metrics,
            "test": test_report,
            "confusion_matrix": test_report["confusion_matrix"],
            "per_class": test_report["per_class"]}
        training_summary = {
            "dataset": dataset_dir.name,
            "trial": trial_name,
            "model_path": str(model_path),
            "epochs_completed": len(history["loss"]),
            "best_validation_loss": best_loss,
            "history": dict(history),
            "validation_metrics": val_metrics,
            "test_metrics": test_metrics,
            "validation_accuracy_report": val_report,
            "test_accuracy_report": test_report,
        }
        # Keep the metrics with the checkpoint as well as in human-readable JSON.
        # This makes every trial model self-describing when copied on its own.
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
        del model, backbone
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    selected_name = max(trial_names, key=lambda x: (
        reports[x]["validation_metrics"]["macro_f1"],
        reports[x]["validation_metrics"]["accuracy"],
        -reports[x]["validation_metrics"]["loss"]))
    selected = reports[selected_name]
    shutil.copy2(out_dir / "trials" / selected_name / "model.pt", out_dir / "model.pt")
    selected["selected_by"], selected["selected_trial"] = "validation_macro_f1", selected_name
    selected["trials"] = {name: {
        "selected": name == selected_name,
        "model_path": str(out_dir / "trials" / name / "model.pt"),
        "training_summary_path": str(
            out_dir / "trials" / name / "training_summary.json"),
        "validation_metrics": report["validation_metrics"],
        "test_metrics": report["test_metrics"],
        "validation_accuracy_report": report["validation"],
        "test_accuracy_report": report["test"]} for name, report in reports.items()}
    comparison = []
    for name, report in reports.items():
        row = {"trial": name, "selected": name == selected_name}
        row.update({f"val_{k}": v for k, v in report["validation_metrics"].items()})
        row.update({f"test_{k}": v for k, v in report["test_metrics"].items()})
        row.update({f"param_{k}": v for k, v in report["strategy"].items()})
        comparison.append(row)
        report["selected"] = name == selected_name
        report["selected_by"] = "validation_macro_f1"
        training_summaries[name]["selected"] = name == selected_name
        training_summaries[name]["selected_by"] = "validation_macro_f1"
        trial_dir = out_dir / "trials" / name
        (trial_dir / "report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        (trial_dir / "training_summary.json").write_text(
            json.dumps(training_summaries[name], indent=2, ensure_ascii=False),
            encoding="utf-8")
    fields = list(dict.fromkeys(k for row in comparison for k in row))
    with (out_dir / "comparison.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader(); writer.writerows(comparison)
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
                "dataset": dataset,
                "trial": trial,
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


def dataset_is_current(directory, expected_clear, args):
    if not (directory / "labels.csv").is_file():
        return False
    try:
        info = json.loads((directory / "dataset_info.json").read_text(encoding="utf-8"))
        if info.get("builder_contract") != REQUIRED_BUILDER_CONTRACT:
            return False
        if not expected_clear:
            return True
        selection = info.get("day22_clear_dino_selection", {})
        return (int(info.get("day22_clear_source_images", 0)) == expected_clear
                and int(info.get("day22_clear_selected_images", 0)) > 0
                and int(info.get("day22_clear_unreadable_images", -1)) == 0
                and float(selection.get("threshold", -1)) == args.dino_threshold
                and int(selection.get("max_per_clip", -1)) == args.dino_max_per_clip)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=CONFIG["input"])
    ap.add_argument("--no-build", action="store_true", default=CONFIG["no_build"])
    ap.add_argument("--rebuild", action="store_true", default=CONFIG["rebuild"])
    ap.add_argument("--stride", type=int, default=CONFIG["stride"])
    ap.add_argument("--jpeg-quality", type=int, default=CONFIG["jpeg_quality"])
    ap.add_argument("--day22-clear-dir", default=CONFIG["day22_clear_dir"])
    ap.add_argument("--dino-threshold", type=float, default=CONFIG["dino_threshold"])
    ap.add_argument("--dino-max-per-clip", type=int, default=CONFIG["dino_max_per_clip"])
    ap.add_argument("--limit", type=int, default=CONFIG["limit"])
    ap.add_argument("--datasets-root", default=CONFIG["datasets_root"])
    ap.add_argument("--datasets", default=CONFIG["datasets"])
    ap.add_argument("--out-root", default=CONFIG["out_root"])
    ap.add_argument("--dino-model", choices=sorted(DINO_MODELS), default=CONFIG["dino_model"])
    ap.add_argument("--img-size", type=int, default=CONFIG["img_size"])
    ap.add_argument("--epochs", type=int, default=CONFIG["epochs"])
    ap.add_argument("--warmup-epochs", type=int, default=CONFIG["warmup_epochs"])
    ap.add_argument("--batch", type=int, default=CONFIG["batch"])
    ap.add_argument("--val-frac", type=float, default=CONFIG["val_frac"])
    ap.add_argument("--test-day", type=int, default=CONFIG["test_day"])
    ap.add_argument("--seed", type=int, default=CONFIG["seed"])
    ap.add_argument("--trials", default=CONFIG["trials"], help=",".join(TRIALS))
    notebook = any("ipykernel" in m or "colab" in m for m in sys.modules)
    args, _ = ap.parse_known_args([] if notebook else None)
    if args.stride < 1 or not 1 <= args.jpeg_quality <= 100:
        ap.error("require stride >= 1 and jpeg-quality between 1 and 100")
    if args.img_size != 224:
        ap.error("pretrained DINOv2 requires --img-size 224")
    if not 0 < args.val_frac < 1:
        ap.error("--val-frac must be between 0 and 1")
    if args.epochs < 1 or not 0 <= args.warmup_epochs <= args.epochs:
        ap.error("require 0 <= --warmup-epochs <= --epochs")
    return args


def main():
    args = parse_args()
    root = Path(args.datasets_root)
    clear = Path(args.day22_clear_dir) if args.day22_clear_dir else None
    expected_clear = (sum(1 for p in clear.glob("*.jpg")
                          if "sem_obstaculos" in p.name.lower())
                      if clear and clear.is_dir() else 0)
    outputs = [root / name for name in pipeline.SCHEMES.values()]
    if args.no_build:
        print("[build] skipped (--no-build)")
    elif all(dataset_is_current(path, expected_clear, args) for path in outputs) \
            and not args.rebuild:
        print(f"[build] skipped: datasets already exist under {root}")
    else:
        pipeline.build_dino_datasets(args)
        gc.collect()

    if args.datasets.strip():
        dirs = [root / name.strip() for name in args.datasets.split(",") if name.strip()]
    else:
        dirs = sorted(path for path in root.iterdir() if (path / "labels.csv").is_file())
    dirs = [path for path in dirs if (path / "labels.csv").is_file()]
    if not dirs:
        raise SystemExit(f"No datasets with labels.csv under {root}")
    summary = {}
    for directory in dirs:
        report = train_one(directory, args.out_root, args)
        if report:
            summary[directory.name] = {
                "selected_trial": report["selected_trial"],
                "selected_by": report["selected_by"],
                "validation_metrics": report["validation_metrics"],
                "test_metrics": report["test_metrics"],
                "trials": report["trials"], "per_class": report["per_class"]}
    Path(args.out_root).mkdir(parents=True, exist_ok=True)
    (Path(args.out_root) / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nALL DONE: {list(summary)}")
    write_accuracy_report(args.out_root, summary)
    return 0


if __name__ == "__main__":
    rc = main()
    # Some Python 3.12 IPython images fail while formatting SystemExit.
    if not any("ipykernel" in m or "colab" in m for m in sys.modules):
        raise SystemExit(rc)
