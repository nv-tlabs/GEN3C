# Open Source License Attribution

   Cosmos uses Open Source components. You can find the details of these open-source projects along with license information below, sorted alphabetically.
   We are grateful to the developers for their contributions to open source and acknowledge these below.

## Better-Profanity - [MIT License](https://github.com/snguyenthanh/better_profanity/blob/master/LICENSE)

   ```

   Copyright (c) 2018 The Python Packaging Authority

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in all
   copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE.

   ```

## FFmpeg - [FFMPEG License](https://github.com/FFmpeg/FFmpeg/blob/master/LICENSE.md)

   ```
   # License

   Most files in FFmpeg are under the GNU Lesser General Public License version 2.1
   or later (LGPL v2.1+). Read the file `COPYING.LGPLv2.1` for details. Some other
   files have MIT/X11/BSD-style licenses. In combination the LGPL v2.1+ applies to
   FFmpeg.

   Some optional parts of FFmpeg are licensed under the GNU General Public License
   version 2 or later (GPL v2+). See the file `COPYING.GPLv2` for details. None of
   these parts are used by default, you have to explicitly pass `--enable-gpl` to
   configure to activate them. In this case, FFmpeg's license changes to GPL v2+.

   Specifically, the GPL parts of FFmpeg are:

   - libpostproc
   - optional x86 optimization in the files
       - `libavcodec/x86/flac_dsp_gpl.asm`
       - `libavcodec/x86/idct_mmx.c`
       - `libavfilter/x86/vf_removegrain.asm`
   - the following building and testing tools
       - `compat/solaris/make_sunver.pl`
       - `doc/t2h.pm`
       - `doc/texi2pod.pl`
       - `libswresample/tests/swresample.c`
       - `tests/checkasm/*`
       - `tests/tiny_ssim.c`
   - the following filters in libavfilter:
       - `signature_lookup.c`
       - `vf_blackframe.c`
       - `vf_boxblur.c`
       - `vf_colormatrix.c`
       - `vf_cover_rect.c`
       - `vf_cropdetect.c`
       - `vf_delogo.c`
       - `vf_eq.c`
       - `vf_find_rect.c`
       - `vf_fspp.c`
       - `vf_histeq.c`
       - `vf_hqdn3d.c`
       - `vf_kerndeint.c`
       - `vf_lensfun.c` (GPL version 3 or later)
       - `vf_mcdeint.c`
       - `vf_mpdecimate.c`
       - `vf_nnedi.c`
       - `vf_owdenoise.c`
       - `vf_perspective.c`
       - `vf_phase.c`
       - `vf_pp.c`
       - `vf_pp7.c`
       - `vf_pullup.c`
       - `vf_repeatfields.c`
       - `vf_sab.c`
       - `vf_signature.c`
       - `vf_smartblur.c`
       - `vf_spp.c`
       - `vf_stereo3d.c`
       - `vf_super2xsai.c`
       - `vf_tinterlace.c`
       - `vf_uspp.c`
       - `vf_vaguedenoiser.c`
       - `vsrc_mptestsrc.c`

   Should you, for whatever reason, prefer to use version 3 of the (L)GPL, then
   the configure parameter `--enable-version3` will activate this licensing option
   for you. Read the file `COPYING.LGPLv3` or, if you have enabled GPL parts,
   `COPYING.GPLv3` to learn the exact legal terms that apply in this case.

   There are a handful of files under other licensing terms, namely:

   * The files `libavcodec/jfdctfst.c`, `libavcodec/jfdctint_template.c` and
     `libavcodec/jrevdct.c` are taken from libjpeg, see the top of the files for
     licensing details. Specifically note that you must credit the IJG in the
     documentation accompanying your program if you only distribute executables.
     You must also indicate any changes including additions and deletions to
     those three files in the documentation.
   * `tests/reference.pnm` is under the expat license.


   ## External libraries

   FFmpeg can be combined with a number of external libraries, which sometimes
   affect the licensing of binaries resulting from the combination.

   ### Compatible libraries

   The following libraries are under GPL version 2:
   - avisynth
   - frei0r
   - libcdio
   - libdavs2
   - librubberband
   - libvidstab
   - libx264
   - libx265
   - libxavs
   - libxavs2
   - libxvid

   When combining them with FFmpeg, FFmpeg needs to be licensed as GPL as well by
   passing `--enable-gpl` to configure.

   The following libraries are under LGPL version 3:
   - gmp
   - libaribb24
   - liblensfun

   When combining them with FFmpeg, use the configure option `--enable-version3` to
   upgrade FFmpeg to the LGPL v3.

   The VMAF, mbedTLS, RK MPI, OpenCORE and VisualOn libraries are under the Apache License
   2.0. That license is incompatible with the LGPL v2.1 and the GPL v2, but not with
   version 3 of those licenses. So to combine these libraries with FFmpeg, the
   license version needs to be upgraded by passing `--enable-version3` to configure.

   The smbclient library is under the GPL v3, to combine it with FFmpeg,
   the options `--enable-gpl` and `--enable-version3` have to be passed to
   configure to upgrade FFmpeg to the GPL v3.

   ### Incompatible libraries

   There are certain libraries you can combine with FFmpeg whose licenses are not
   compatible with the GPL and/or the LGPL. If you wish to enable these
   libraries, even in circumstances that their license may be incompatible, pass
   `--enable-nonfree` to configure. This will cause the resulting binary to be
   unredistributable.

   The Fraunhofer FDK AAC and OpenSSL libraries are under licenses which are
   incompatible with the GPLv2 and v3. To the best of our knowledge, they are
   compatible with the LGPL.

   ```

## Hydra-core [MIT License](https://github.com/facebookresearch/hydra/blob/main/LICENSE)

   ```

   MIT License

   Copyright (c) Facebook, Inc. and its affiliates.

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in all
   copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE.

   ```

## Llama-Guard-3-8B [META LLAMA 3 COMMUNITY LICENSE](https://github.com/meta-llama/llama3/blob/main/LICENSE)

   ```

   META LLAMA 3 COMMUNITY LICENSE AGREEMENT

   Meta Llama 3 Version Release Date: April 18, 2024

   “Agreement” means the terms and conditions for use, reproduction, distribution, and
   modification of the Llama Materials set forth herein.

   “Documentation” means the specifications, manuals, and documentation accompanying Meta
   Llama 3 distributed by Meta at https://llama.meta.com/get-started/.

   “Licensee” or “you” means you, or your employer or any other person or entity (if you are
   entering into this Agreement on such person or entity’s behalf), of the age required under
   applicable laws, rules, or regulations to provide legal consent and that has legal authority
   to bind your employer or such other person or entity if you are entering into this Agreement
   on their behalf.

   “Meta Llama 3” means the foundational large language models and software and algorithms,
   including machine-learning model code, trained model weights, inference-enabling code,
   training-enabling code, fine-tuning-enabling code, and other elements of the foregoing
   distributed by Meta at https://llama.meta.com/llama-downloads.

   “Llama Materials” means, collectively, Meta’s proprietary Meta Llama 3 and Documentation
   (and any portion thereof) made available under this Agreement.

   “Meta” or “we” means Meta Platforms Ireland Limited (if you are located in or, if you are
   an entity, your principal place of business is in the EEA or Switzerland) and Meta Platforms,
   Inc. (if you are located outside of the EEA or Switzerland).

   By clicking “I Accept” below or by using or distributing any portion or element of the Llama
   Materials, you agree to be bound by this Agreement.

   1. License Rights and Redistribution.

   a. Grant of Rights. You are granted a non-exclusive, worldwide, non-transferable and
   royalty-free limited license under Meta’s intellectual property or other rights owned by
   Meta embodied in the Llama Materials to use, reproduce, distribute, copy, create derivative
   works of, and make modifications to the Llama Materials.

   b. Redistribution and Use.
      i. If you distribute or make available the Llama Materials (or any derivative works
      thereof), or a product or service that uses any of them, including another AI model, you
      shall (A) provide a copy of this Agreement with any such Llama Materials; and (B)
      prominently display “Built with Meta Llama 3” on a related website, user interface,
      blogpost, about page, or product documentation. If you use the Llama Materials to create,
      train, fine tune, or otherwise improve an AI model, which is distributed or made available,
      you shall also include “Llama 3” at the beginning of any such AI model name.

      ii. If you receive Llama Materials, or any derivative works thereof, from a Licensee as
      part of an integrated end user product, then Section 2 of this Agreement will not apply
      to you.

      iii. You must retain in all copies of the Llama Materials that you distribute the
      following attribution notice within a “Notice” text file distributed as a part of such
      copies: “Meta Llama 3 is licensed under the Meta Llama 3 Community License, Copyright ©
      Meta Platforms, Inc. All Rights Reserved.”

      iv. Your use of the Llama Materials must comply with applicable laws and regulations
      (including trade compliance laws and regulations) and adhere to the Acceptable Use Policy
      for the Llama Materials (available at https://llama.meta.com/llama3/use-policy), which
      is hereby incorporated by reference into this Agreement.

      v. You will not use the Llama Materials or any output or results of the Llama Materials
      to improve any other large language model (excluding Meta Llama 3 or derivative works
      thereof).

   2. Additional Commercial Terms.

   If, on the Meta Llama 3 version release date, the monthly active users of the products or
   services made available by or for Licensee, or Licensee’s affiliates, is greater than 700
   million monthly active users in the preceding calendar month, you must request a license
   from Meta, which Meta may grant to you in its sole discretion, and you are not authorized
   to exercise any of the rights under this Agreement unless or until Meta otherwise expressly
   grants you such rights.

   3. Disclaimer of Warranty.

   UNLESS REQUIRED BY APPLICABLE LAW, THE LLAMA MATERIALS AND ANY OUTPUT AND RESULTS THEREFROM
   ARE PROVIDED ON AN “AS IS” BASIS, WITHOUT WARRANTIES OF ANY KIND, AND META DISCLAIMS ALL
   WARRANTIES OF ANY KIND, BOTH EXPRESS AND IMPLIED, INCLUDING, WITHOUT LIMITATION, ANY
   WARRANTIES OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
   YOU ARE SOLELY RESPONSIBLE FOR DETERMINING THE APPROPRIATENESS OF USING OR REDISTRIBUTING
   THE LLAMA MATERIALS AND ASSUME ANY RISKS ASSOCIATED WITH YOUR USE OF THE LLAMA MATERIALS
   AND ANY OUTPUT AND RESULTS.

   4. Limitation of Liability.

   IN NO EVENT WILL META OR ITS AFFILIATES BE LIABLE UNDER ANY THEORY OF LIABILITY, WHETHER IN
   CONTRACT, TORT, NEGLIGENCE, PRODUCTS LIABILITY, OR OTHERWISE, ARISING OUT OF THIS AGREEMENT,
   FOR ANY LOST PROFITS OR ANY INDIRECT, SPECIAL, CONSEQUENTIAL, INCIDENTAL, EXEMPLARY OR
   PUNITIVE DAMAGES, EVEN IF META OR ITS AFFILIATES HAVE BEEN ADVISED OF THE POSSIBILITY OF ANY
   OF THE FOREGOING.

   5. Intellectual Property.

   a. No trademark licenses are granted under this Agreement, and in connection with the Llama
   Materials, neither Meta nor Licensee may use any name or mark owned by or associated with
   the other or any of its affiliates, except as required for reasonable and customary use in
   describing and redistributing the Llama Materials or as set forth in this Section 5(a).
   Meta hereby grants you a license to use “Llama 3” (the “Mark”) solely as required to comply
   with the last sentence of Section 1.b.i. You will comply with Meta’s brand guidelines
   (currently accessible at https://about.meta.com/brand/resources/meta/company-brand/).
   All goodwill arising out of your use of the Mark will inure to the benefit of Meta.

   b. Subject to Meta’s ownership of Llama Materials and derivatives made by or for Meta, with
   respect to any derivative works and modifications of the Llama Materials that are made by
   you, as between you and Meta, you are and will be the owner of such derivative works and
   modifications.

   c. If you institute litigation or other proceedings against Meta or any entity (including a
   cross-claim or counterclaim in a lawsuit) alleging that the Llama Materials or Meta Llama 3
   outputs or results, or any portion of any of the foregoing, constitutes infringement of
   intellectual property or other rights owned or licensable by you, then any licenses granted
   to you under this Agreement shall terminate as of the date such litigation or claim is filed
   or instituted. You will indemnify and hold harmless Meta from and against any claim by any
   third party arising out of or related to your use or distribution of the Llama Materials.

   6. Term and Termination.

   The term of this Agreement will commence upon your acceptance of this Agreement or access
   to the Llama Materials and will continue in full force and effect until terminated in
   accordance with the terms and conditions herein. Meta may terminate this Agreement if you
   are in breach of any term or condition of this Agreement. Upon termination of this Agreement,
   you shall delete and cease use of the Llama Materials. Sections 3, 4, and 7 shall survive
   the termination of this Agreement.

   7. Governing Law and Jurisdiction.

   This Agreement will be governed and construed under the laws of the State of California
   without regard to choice of law principles, and the UN Convention on Contracts for the
   International Sale of Goods does not apply to this Agreement. The courts of California
   shall have exclusive jurisdiction of any dispute arising out of this Agreement.

   META LLAMA 3 ACCEPTABLE USE POLICY

   Meta is committed to promoting safe and fair use of its tools and features, including Meta
   Llama 3. If you access or use Meta Llama 3, you agree to this Acceptable Use Policy
   (“Policy”). The most recent copy of this policy can be found at
   https://llama.meta.com/llama3/use-policy.

   Prohibited Uses

   We want everyone to use Meta Llama 3 safely and responsibly. You agree you will not use, or
   allow others to use, Meta Llama 3 to:

   1. Violate the law or others’ rights, including to:

   a. Engage in, promote, generate, contribute to, encourage, plan, incite, or further illegal
   or unlawful activity or content, such as:

      i. Violence or terrorism
      ii. Exploitation or harm to children, including the solicitation, creation, acquisition,
      or dissemination of child exploitative content or failure to report Child Sexual Abuse
      Material
      iii. Human trafficking, exploitation, and sexual violence
      iv. The illegal distribution of information or materials to minors, including obscene
      materials, or failure to employ legally required age-gating in connection with such
      information or materials
      v. Sexual solicitation
      vi. Any other criminal activity

   b. Engage in, promote, incite, or facilitate the harassment, abuse, threatening, or
   bullying of individuals or groups of individuals

   c. Engage in, promote, incite, or facilitate discrimination or other unlawful or harmful
   conduct in the provision of employment, employment benefits, credit, housing, other economic
   benefits, or other essential goods and services

   d. Engage in the unauthorized or unlicensed practice of any profession including, but not
   limited to, financial, legal, medical/health, or related professional practices

   e. Collect, process, disclose, generate, or infer health, demographic, or other sensitive
   personal or private information about individuals without rights and consents required by
   applicable laws

   f. Engage in or facilitate any action or generate any content that infringes, misappropriates,
   or otherwise violates any third-party rights, including the outputs or results of any
   products or services using the Llama Materials

   g. Create, generate, or facilitate the creation of malicious code, malware, computer viruses
   or do anything else that could disable, overburden, interfere with or impair the proper
   working, integrity, operation, or appearance of a website or computer system

   2. Engage in, promote, incite, facilitate, or assist in the planning or development of
   activities that present a risk of death or bodily harm to individuals, including use of Meta
   Llama 3 related to the following:

   a. Military, warfare, nuclear industries or applications, espionage, use for materials or
   activities that are subject to the International Traffic Arms Regulations (ITAR) maintained
   by the United States Department of State
   b. Guns and illegal weapons (including weapon development)
   c. Illegal drugs and regulated/controlled substances
   d. Operation of critical infrastructure, transportation technologies, or heavy machinery
   e. Self-harm or harm to others, including suicide, cutting, and eating disorders
   f. Any content intended to incite or promote violence, abuse, or any infliction of bodily
   harm to an individual

   3. Intentionally deceive or mislead others, including use of Meta Llama 3 related to the
   following:

   a. Generating, promoting, or furthering fraud or the creation or promotion of disinformation
   b. Generating, promoting, or furthering defamatory content, including the creation of
   defamatory statements, images, or other content
   c. Generating, promoting, or further distributing spam
   d. Impersonating another individual without consent, authorization, or legal right
   e. Representing that the use of Meta Llama 3 or outputs are human-generated
   f. Generating or facilitating false online engagement, including fake reviews and other
   means of fake online engagement
   g. Fail to appropriately disclose to end users any known dangers of your AI system

   Please report any violation of this Policy, software “bug,” or other problems that could
   lead to a violation of this Policy through one of the following means:

   * Reporting issues with the model: https://github.com/meta-llama/llama3
   * Reporting risky content generated by the model: developers.facebook.com/llama_output_feedback
   * Reporting bugs and security concerns: facebook.com/whitehat/info
   * Reporting violations of the Acceptable Use Policy or unlicensed uses of Meta Llama 3:
      LlamaUseReport@meta.com

   ```

## ImageIo - [BSD 2-Clause "Simplified" License](https://github.com/imageio/imageio/blob/master/LICENSE)

   ```

   Copyright (c) 2014-2022, imageio developers
   All rights reserved.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are met:

   * Redistributions of source code must retain the above copyright notice, this
     list of conditions and the following disclaimer.

   * Redistributions in binary form must reproduce the above copyright notice,
     this list of conditions and the following disclaimer in the documentation
     and/or other materials provided with the distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
   AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
   IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
   DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
   FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
   DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
   SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
   CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
   OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

   ```

## Iopath - [MIT License](https://github.com/facebookresearch/iopath/blob/main/LICENSE)

   ```
   MIT License

   Copyright (c) Facebook, Inc. and its affiliates.

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in all
   copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE.

   ```

## Loguru - [MIT License](https://github.com/Delgan/loguru/blob/master/LICENSE)

   ```

   MIT License

   Copyright (c) 2017

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in all
   copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE.

   ```

## Mediapy - [Apache License 2.0](https://github.com/google/mediapy/blob/main/LICENSE)

   ```

                                    Apache License
                              Version 2.0, January 2004
                           http://www.apache.org/licenses/

      TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

      1. Definitions.

         "License" shall mean the terms and conditions for use, reproduction,
         and distribution as defined by Sections 1 through 9 of this document.

         "Licensor" shall mean the copyright owner or entity authorized by
         the copyright owner that is granting the License.

         "Legal Entity" shall mean the union of the acting entity and all
         other entities that control, are controlled by, or are under common
         control with that entity. For the purposes of this definition,
         "control" means (i) the power, direct or indirect, to cause the
         direction or management of such entity, whether by contract or
         otherwise, or (ii) ownership of fifty percent (50%) or more of the
         outstanding shares, or (iii) beneficial ownership of such entity.

         "You" (or "Your") shall mean an individual or Legal Entity
         exercising permissions granted by this License.

         "Source" form shall mean the preferred form for making modifications,
         including but not limited to software source code, documentation
         source, and configuration files.

         "Object" form shall mean any form resulting from mechanical
         transformation or translation of a Source form, including but
         not limited to compiled object code, generated documentation,
         and conversions to other media types.

         "Work" shall mean the work of authorship, whether in Source or
         Object form, made available under the License, as indicated by a
         copyright notice that is included in or attached to the work
         (an example is provided in the Appendix below).

         "Derivative Works" shall mean any work, whether in Source or Object
         form, that is based on (or derived from) the Work and for which the
         editorial revisions, annotations, elaborations, or other modifications
         represent, as a whole, an original work of authorship. For the purposes
         of this License, Derivative Works shall not include works that remain
         separable from, or merely link (or bind by name) to the interfaces of,
         the Work and Derivative Works thereof.

         "Contribution" shall mean any work of authorship, including
         the original version of the Work and any modifications or additions
         to that Work or Derivative Works thereof, that is intentionally
         submitted to Licensor for inclusion in the Work by the copyright owner
         or by an individual or Legal Entity authorized to submit on behalf of
         the copyright owner. For the purposes of this definition, "submitted"
         means any form of electronic, verbal, or written communication sent
         to the Licensor or its representatives, including but not limited to
         communication on electronic mailing lists, source code control systems,
         and issue tracking systems that are managed by, or on behalf of, the
         Licensor for the purpose of discussing and improving the Work, but
         excluding communication that is conspicuously marked or otherwise
         designated in writing by the copyright owner as "Not a Contribution."

         "Contributor" shall mean Licensor and any individual or Legal Entity
         on behalf of whom a Contribution has been received by Licensor and
         subsequently incorporated within the Work.

      2. Grant of Copyright License. Subject to the terms and conditions of
         this License, each Contributor hereby grants to You a perpetual,
         worldwide, non-exclusive, no-charge, royalty-free, irrevocable
         copyright license to reproduce, prepare Derivative Works of,
         publicly display, publicly perform, sublicense, and distribute the
         Work and such Derivative Works in Source or Object form.

      3. Grant of Patent License. Subject to the terms and conditions of
         this License, each Contributor hereby grants to You a perpetual,
         worldwide, non-exclusive, no-charge, royalty-free, irrevocable
         (except as stated in this section) patent license to make, have made,
         use, offer to sell, sell, import, and otherwise transfer the Work,
         where such license applies only to those patent claims licensable
         by such Contributor that are necessarily infringed by their
         Contribution(s) alone or by combination of their Contribution(s)
         with the Work to which such Contribution(s) was submitted. If You
         institute patent litigation against any entity (including a
         cross-claim or counterclaim in a lawsuit) alleging that the Work
         or a Contribution incorporated within the Work constitutes direct
         or contributory patent infringement, then any patent licenses
         granted to You under this License for that Work shall terminate
         as of the date such litigation is filed.

      4. Redistribution. You may reproduce and distribute copies of the
         Work or Derivative Works thereof in any medium, with or without
         modifications, and in Source or Object form, provided that You
         meet the following conditions:

         (a) You must give any other recipients of the Work or
             Derivative Works a copy of this License; and

         (b) You must cause any modified files to carry prominent notices
             stating that You changed the files; and

         (c) You must retain, in the Source form of any Derivative Works
             that You distribute, all copyright, patent, trademark, and
             attribution notices from the Source form of the Work,
             excluding those notices that do not pertain to any part of
             the Derivative Works; and

         (d) If the Work includes a "NOTICE" text file as part of its
             distribution, then any Derivative Works that You distribute must
             include a readable copy of the attribution notices contained
             within such NOTICE file, excluding those notices that do not
             pertain to any part of the Derivative Works, in at least one
             of the following places: within a NOTICE text file distributed
             as part of the Derivative Works; within the Source form or
             documentation, if provided along with the Derivative Works; or,
             within a display generated by the Derivative Works, if and
             wherever such third-party notices normally appear. The contents
             of the NOTICE file are for informational purposes only and
             do not modify the License. You may add Your own attribution
             notices within Derivative Works that You distribute, alongside
             or as an addendum to the NOTICE text from the Work, provided
             that such additional attribution notices cannot be construed
             as modifying the License.

         You may add Your own copyright statement to Your modifications and
         may provide additional or different license terms and conditions
         for use, reproduction, or distribution of Your modifications, or
         for any such Derivative Works as a whole, provided Your use,
         reproduction, and distribution of the Work otherwise complies with
         the conditions stated in this License.

      5. Submission of Contributions. Unless You explicitly state otherwise,
         any Contribution intentionally submitted for inclusion in the Work
         by You to the Licensor shall be under the terms and conditions of
         this License, without any additional terms or conditions.
         Notwithstanding the above, nothing herein shall supersede or modify
         the terms of any separate license agreement you may have executed
         with Licensor regarding such Contributions.

      6. Trademarks. This License does not grant permission to use the trade
         names, trademarks, service marks, or product names of the Licensor,
         except as required for reasonable and customary use in describing the
         origin of the Work and reproducing the content of the NOTICE file.

      7. Disclaimer of Warranty. Unless required by applicable law or
         agreed to in writing, Licensor provides the Work (and each
         Contributor provides its Contributions) on an "AS IS" BASIS,
         WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
         implied, including, without limitation, any warranties or conditions
         of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
         PARTICULAR PURPOSE. You are solely responsible for determining the
         appropriateness of using or redistributing the Work and assume any
         risks associated with Your exercise of permissions under this License.

      8. Limitation of Liability. In no event and under no legal theory,
         whether in tort (including negligence), contract, or otherwise,
         unless required by applicable law (such as deliberate and grossly
         negligent acts) or agreed to in writing, shall any Contributor be
         liable to You for damages, including any direct, indirect, special,
         incidental, or consequential damages of any character arising as a
         result of this License or out of the use or inability to use the
         Work (including but not limited to damages for loss of goodwill,
         work stoppage, computer failure or malfunction, or any and all
         other commercial damages or losses), even if such Contributor
         has been advised of the possibility of such damages.

      9. Accepting Warranty or Additional Liability. While redistributing
         the Work or Derivative Works thereof, You may choose to offer,
         and charge a fee for, acceptance of support, warranty, indemnity,
         or other liability obligations and/or rights consistent with this
         License. However, in accepting such obligations, You may act only
         on Your own behalf and on Your sole responsibility, not on behalf
         of any other Contributor, and only if You agree to indemnify,
         defend, and hold each Contributor harmless for any liability
         incurred by, or claims asserted against, such Contributor by reason
         of your accepting any such warranty or additional liability.

      END OF TERMS AND CONDITIONS

      APPENDIX: How to apply the Apache License to your work.

         To apply the Apache License to your work, attach the following
         boilerplate notice, with the fields enclosed by brackets "[]"
         replaced with your own identifying information. (Don't include
         the brackets!)  The text should be enclosed in the appropriate
         comment syntax for the file format. We also recommend that a
         file or class name and description of purpose be included on the
         same "printed page" as the copyright notice for easier
         identification within third-party archives.

      Copyright [yyyy] [name of copyright owner]

      Licensed under the Apache License, Version 2.0 (the "License");
      you may not use this file except in compliance with the License.
      You may obtain a copy of the License at

          http://www.apache.org/licenses/LICENSE-2.0

      Unless required by applicable law or agreed to in writing, software
      distributed under the License is distributed on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
      See the License for the specific language governing permissions and
      limitations under the License.

   ```

## Nltk - [Apache License 2.0](https://github.com/nltk/nltk/blob/develop/LICENSE.txt)

   ```

                                    Apache License
                              Version 2.0, January 2004
                           http://www.apache.org/licenses/

      TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

      1. Definitions.

         "License" shall mean the terms and conditions for use, reproduction,
         and distribution as defined by Sections 1 through 9 of this document.

         "Licensor" shall mean the copyright owner or entity authorized by
         the copyright owner that is granting the License.

         "Legal Entity" shall mean the union of the acting entity and all
         other entities that control, are controlled by, or are under common
         control with that entity. For the purposes of this definition,
         "control" means (i) the power, direct or indirect, to cause the
         direction or management of such entity, whether by contract or
         otherwise, or (ii) ownership of fifty percent (50%) or more of the
         outstanding shares, or (iii) beneficial ownership of such entity.

         "You" (or "Your") shall mean an individual or Legal Entity
         exercising permissions granted by this License.

         "Source" form shall mean the preferred form for making modifications,
         including but not limited to software source code, documentation
         source, and configuration files.

         "Object" form shall mean any form resulting from mechanical
         transformation or translation of a Source form, including but
         not limited to compiled object code, generated documentation,
         and conversions to other media types.

         "Work" shall mean the work of authorship, whether in Source or
         Object form, made available under the License, as indicated by a
         copyright notice that is included in or attached to the work
         (an example is provided in the Appendix below).

         "Derivative Works" shall mean any work, whether in Source or Object
         form, that is based on (or derived from) the Work and for which the
         editorial revisions, annotations, elaborations, or other modifications
         represent, as a whole, an original work of authorship. For the purposes
         of this License, Derivative Works shall not include works that remain
         separable from, or merely link (or bind by name) to the interfaces of,
         the Work and Derivative Works thereof.

         "Contribution" shall mean any work of authorship, including
         the original version of the Work and any modifications or additions
         to that Work or Derivative Works thereof, that is intentionally
         submitted to Licensor for inclusion in the Work by the copyright owner
         or by an individual or Legal Entity authorized to submit on behalf of
         the copyright owner. For the purposes of this definition, "submitted"
         means any form of electronic, verbal, or written communication sent
         to the Licensor or its representatives, including but not limited to
         communication on electronic mailing lists, source code control systems,
         and issue tracking systems that are managed by, or on behalf of, the
         Licensor for the purpose of discussing and improving the Work, but
         excluding communication that is conspicuously marked or otherwise
         designated in writing by the copyright owner as "Not a Contribution."

         "Contributor" shall mean Licensor and any individual or Legal Entity
         on behalf of whom a Contribution has been received by Licensor and
         subsequently incorporated within the Work.

      2. Grant of Copyright License. Subject to the terms and conditions of
         this License, each Contributor hereby grants to You a perpetual,
         worldwide, non-exclusive, no-charge, royalty-free, irrevocable
         copyright license to reproduce, prepare Derivative Works of,
         publicly display, publicly perform, sublicense, and distribute the
         Work and such Derivative Works in Source or Object form.

      3. Grant of Patent License. Subject to the terms and conditions of
         this License, each Contributor hereby grants to You a perpetual,
         worldwide, non-exclusive, no-charge, royalty-free, irrevocable
         (except as stated in this section) patent license to make, have made,
         use, offer to sell, sell, import, and otherwise transfer the Work,
         where such license applies only to those patent claims licensable
         by such Contributor that are necessarily infringed by their
         Contribution(s) alone or by combination of their Contribution(s)
         with the Work to which such Contribution(s) was submitted. If You
         institute patent litigation against any entity (including a
         cross-claim or counterclaim in a lawsuit) alleging that the Work
         or a Contribution incorporated within the Work constitutes direct
         or contributory patent infringement, then any patent licenses
         granted to You under this License for that Work shall terminate
         as of the date such litigation is filed.

      4. Redistribution. You may reproduce and distribute copies of the
         Work or Derivative Works thereof in any medium, with or without
         modifications, and in Source or Object form, provided that You
         meet the following conditions:

         (a) You must give any other recipients of the Work or
             Derivative Works a copy of this License; and

         (b) You must cause any modified files to carry prominent notices
             stating that You changed the files; and

         (c) You must retain, in the Source form of any Derivative Works
             that You distribute, all copyright, patent, trademark, and
             attribution notices from the Source form of the Work,
             excluding those notices that do not pertain to any part of
             the Derivative Works; and

         (d) If the Work includes a "NOTICE" text file as part of its
             distribution, then any Derivative Works that You distribute must
             include a readable copy of the attribution notices contained
             within such NOTICE file, excluding those notices that do not
             pertain to any part of the Derivative Works, in at least one
             of the following places: within a NOTICE text file distributed
             as part of the Derivative Works; within the Source form or
             documentation, if provided along with the Derivative Works; or,
             within a display generated by the Derivative Works, if and
             wherever such third-party notices normally appear. The contents
             of the NOTICE file are for informational purposes only and
             do not modify the License. You may add Your own attribution
             notices within Derivative Works that You distribute, alongside
             or as an addendum to the NOTICE text from the Work, provided
             that such additional attribution notices cannot be construed
             as modifying the License.

         You may add Your own copyright statement to Your modifications and
         may provide additional or different license terms and conditions
         for use, reproduction, or distribution of Your modifications, or
         for any such Derivative Works as a whole, provided Your use,
         reproduction, and distribution of the Work otherwise complies with
         the conditions stated in this License.

      5. Submission of Contributions. Unless You explicitly state otherwise,
         any Contribution intentionally submitted for inclusion in the Work
         by You to the Licensor shall be under the terms and conditions of
         this License, without any additional terms or conditions.
         Notwithstanding the above, nothing herein shall supersede or modify
         the terms of any separate license agreement you may have executed
         with Licensor regarding such Contributions.

      6. Trademarks. This License does not grant permission to use the trade
         names, trademarks, service marks, or product names of the Licensor,
         except as required for reasonable and customary use in describing the
         origin of the Work and reproducing the content of the NOTICE file.

      7. Disclaimer of Warranty. Unless required by applicable law or
         agreed to in writing, Licensor provides the Work (and each
         Contributor provides its Contributions) on an "AS IS" BASIS,
         WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
         implied, including, without limitation, any warranties or conditions
         of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
         PARTICULAR PURPOSE. You are solely responsible for determining the
         appropriateness of using or redistributing the Work and assume any
         risks associated with Your exercise of permissions under this License.

      8. Limitation of Liability. In no event and under no legal theory,
         whether in tort (including negligence), contract, or otherwise,
         unless required by applicable law (such as deliberate and grossly
         negligent acts) or agreed to in writing, shall any Contributor be
         liable to You for damages, including any direct, indirect, special,
         incidental, or consequential damages of any character arising as a
         result of this License or out of the use or inability to use the
         Work (including but not limited to damages for loss of goodwill,
         work stoppage, computer failure or malfunction, or any and all
         other commercial damages or losses), even if such Contributor
         has been advised of the possibility of such damages.

      9. Accepting Warranty or Additional Liability. While redistributing
         the Work or Derivative Works thereof, You may choose to offer,
         and charge a fee for, acceptance of support, warranty, indemnity,
         or other liability obligations and/or rights consistent with this
         License. However, in accepting such obligations, You may act only
         on Your own behalf and on Your sole responsibility, not on behalf
         of any other Contributor, and only if You agree to indemnify,
         defend, and hold each Contributor harmless for any liability
         incurred by, or claims asserted against, such Contributor by reason
         of your accepting any such warranty or additional liability.

      END OF TERMS AND CONDITIONS

      APPENDIX: How to apply the Apache License to your work.

         To apply the Apache License to your work, attach the following
         boilerplate notice, with the fields enclosed by brackets "[]"
         replaced with your own identifying information. (Don't include
         the brackets!)  The text should be enclosed in the appropriate
         comment syntax for the file format. We also recommend that a
         file or class name and description of purpose be included on the
         same "printed page" as the copyright notice for easier
         identification within third-party archives.

      Copyright [yyyy] [name of copyright owner]

      Licensed under the Apache License, Version 2.0 (the "License");
      you may not use this file except in compliance with the License.
      You may obtain a copy of the License at

          http://www.apache.org/licenses/LICENSE-2.0

      Unless required by applicable law or agreed to in writing, software
      distributed under the License is distributed on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
      See the License for the specific language governing permissions and
      limitations under the License.

   ```

## PEFT - [Apache License 2.0](https://github.com/huggingface/peft/blob/main/LICENSE)

   ```

                                    Apache License
                              Version 2.0, January 2004
                           http://www.apache.org/licenses/

      TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

      1. Definitions.

         "License" shall mean the terms and conditions for use, reproduction,
         and distribution as defined by Sections 1 through 9 of this document.

         "Licensor" shall mean the copyright owner or entity authorized by
         the copyright owner that is granting the License.

         "Legal Entity" shall mean the union of the acting entity and all
         other entities that control, are controlled by, or are under common
         control with that entity. For the purposes of this definition,
         "control" means (i) the power, direct or indirect, to cause the
         direction or management of such entity, whether by contract or
         otherwise, or (ii) ownership of fifty percent (50%) or more of the
         outstanding shares, or (iii) beneficial ownership of such entity.

         "You" (or "Your") shall mean an individual or Legal Entity
         exercising permissions granted by this License.

         "Source" form shall mean the preferred form for making modifications,
         including but not limited to software source code, documentation
         source, and configuration files.

         "Object" form shall mean any form resulting from mechanical
         transformation or translation of a Source form, including but
         not limited to compiled object code, generated documentation,
         and conversions to other media types.

         "Work" shall mean the work of authorship, whether in Source or
         Object form, made available under the License, as indicated by a
         copyright notice that is included in or attached to the work
         (an example is provided in the Appendix below).

         "Derivative Works" shall mean any work, whether in Source or Object
         form, that is based on (or derived from) the Work and for which the
         editorial revisions, annotations, elaborations, or other modifications
         represent, as a whole, an original work of authorship. For the purposes
         of this License, Derivative Works shall not include works that remain
         separable from, or merely link (or bind by name) to the interfaces of,
         the Work and Derivative Works thereof.

         "Contribution" shall mean any work of authorship, including
         the original version of the Work and any modifications or additions
         to that Work or Derivative Works thereof, that is intentionally
         submitted to Licensor for inclusion in the Work by the copyright owner
         or by an individual or Legal Entity authorized to submit on behalf of
         the copyright owner. For the purposes of this definition, "submitted"
         means any form of electronic, verbal, or written communication sent
         to the Licensor or its representatives, including but not limited to
         communication on electronic mailing lists, source code control systems,
         and issue tracking systems that are managed by, or on behalf of, the
         Licensor for the purpose of discussing and improving the Work, but
         excluding communication that is conspicuously marked or otherwise
         designated in writing by the copyright owner as "Not a Contribution."

         "Contributor" shall mean Licensor and any individual or Legal Entity
         on behalf of whom a Contribution has been received by Licensor and
         subsequently incorporated within the Work.

      2. Grant of Copyright License. Subject to the terms and conditions of
         this License, each Contributor hereby grants to You a perpetual,
         worldwide, non-exclusive, no-charge, royalty-free, irrevocable
         copyright license to reproduce, prepare Derivative Works of,
         publicly display, publicly perform, sublicense, and distribute the
         Work and such Derivative Works in Source or Object form.

      3. Grant of Patent License. Subject to the terms and conditions of
         this License, each Contributor hereby grants to You a perpetual,
         worldwide, non-exclusive, no-charge, royalty-free, irrevocable
         (except as stated in this section) patent license to make, have made,
         use, offer to sell, sell, import, and otherwise transfer the Work,
         where such license applies only to those patent claims licensable
         by such Contributor that are necessarily infringed by their
         Contribution(s) alone or by combination of their Contribution(s)
         with the Work to which such Contribution(s) was submitted. If You
         institute patent litigation against any entity (including a
         cross-claim or counterclaim in a lawsuit) alleging that the Work
         or a Contribution incorporated within the Work constitutes direct
         or contributory patent infringement, then any patent licenses
         granted to You under this License for that Work shall terminate
         as of the date such litigation is filed.

      4. Redistribution. You may reproduce and distribute copies of the
         Work or Derivative Works thereof in any medium, with or without
         modifications, and in Source or Object form, provided that You
         meet the following conditions:

         (a) You must give any other recipients of the Work or
             Derivative Works a copy of this License; and

         (b) You must cause any modified files to carry prominent notices
             stating that You changed the files; and

         (c) You must retain, in the Source form of any Derivative Works
             that You distribute, all copyright, patent, trademark, and
             attribution notices from the Source form of the Work,
             excluding those notices that do not pertain to any part of
             the Derivative Works; and

         (d) If the Work includes a "NOTICE" text file as part of its
             distribution, then any Derivative Works that You distribute must
             include a readable copy of the attribution notices contained
             within such NOTICE file, excluding those notices that do not
             pertain to any part of the Derivative Works, in at least one
             of the following places: within a NOTICE text file distributed
             as part of the Derivative Works; within the Source form or
             documentation, if provided along with the Derivative Works; or,
             within a display generated by the Derivative Works, if and
             wherever such third-party notices normally appear. The contents
             of the NOTICE file are for informational purposes only and
             do not modify the License. You may add Your own attribution
             notices within Derivative Works that You distribute, alongside
             or as an addendum to the NOTICE text from the Work, provided
             that such additional attribution notices cannot be construed
             as modifying the License.

         You may add Your own copyright statement to Your modifications and
         may provide additional or different license terms and conditions
         for use, reproduction, or distribution of Your modifications, or
         for any such Derivative Works as a whole, provided Your use,
         reproduction, and distribution of the Work otherwise complies with
         the conditions stated in this License.

      5. Submission of Contributions. Unless You explicitly state otherwise,
         any Contribution intentionally submitted for inclusion in the Work
         by You to the Licensor shall be under the terms and conditions of
         this License, without any additional terms or conditions.
         Notwithstanding the above, nothing herein shall supersede or modify
         the terms of any separate license agreement you may have executed
         with Licensor regarding such Contributions.

      6. Trademarks. This License does not grant permission to use the trade
         names, trademarks, service marks, or product names of the Licensor,
         except as required for reasonable and customary use in describing the
         origin of the Work and reproducing the content of the NOTICE file.

      7. Disclaimer of Warranty. Unless required by applicable law or
         agreed to in writing, Licensor provides the Work (and each
         Contributor provides its Contributions) on an "AS IS" BASIS,
         WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
         implied, including, without limitation, any warranties or conditions
         of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
         PARTICULAR PURPOSE. You are solely responsible for determining the
         appropriateness of using or redistributing the Work and assume any
         risks associated with Your exercise of permissions under this License.

      8. Limitation of Liability. In no event and under no legal theory,
         whether in tort (including negligence), contract, or otherwise,
         unless required by applicable law (such as deliberate and grossly
         negligent acts) or agreed to in writing, shall any Contributor be
         liable to You for damages, including any direct, indirect, special,
         incidental, or consequential damages of any character arising as a
         result of this License or out of the use or inability to use the
         Work (including but not limited to damages for loss of goodwill,
         work stoppage, computer failure or malfunction, or any and all
         other commercial damages or losses), even if such Contributor
         has been advised of the possibility of such damages.

      9. Accepting Warranty or Additional Liability. While redistributing
         the Work or Derivative Works thereof, You may choose to offer,
         and charge a fee for, acceptance of support, warranty, indemnity,
         or other liability obligations and/or rights consistent with this
         License. However, in accepting such obligations, You may act only
         on Your own behalf and on Your sole responsibility, not on behalf
         of any other Contributor, and only if You agree to indemnify,
         defend, and hold each Contributor harmless for any liability
         incurred by, or claims asserted against, such Contributor by reason
         of your accepting any such warranty or additional liability.

      END OF TERMS AND CONDITIONS

      APPENDIX: How to apply the Apache License to your work.

         To apply the Apache License to your work, attach the following
         boilerplate notice, with the fields enclosed by brackets "[]"
         replaced with your own identifying information. (Don't include
         the brackets!)  The text should be enclosed in the appropriate
         comment syntax for the file format. We also recommend that a
         file or class name and description of purpose be included on the
         same "printed page" as the copyright notice for easier
         identification within third-party archives.

      Copyright [yyyy] [name of copyright owner]

      Licensed under the Apache License, Version 2.0 (the "License");
      you may not use this file except in compliance with the License.
      You may obtain a copy of the License at

          http://www.apache.org/licenses/LICENSE-2.0

      Unless required by applicable law or agreed to in writing, software
      distributed under the License is distributed on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
      See the License for the specific language governing permissions and
      limitations under the License.

   ```

## Pillow - [MIT License](https://github.com/python-pillow/Pillow/blob/main/LICENSE)

   ```

   The Python Imaging Library (PIL) is

       Copyright © 1997-2011 by Secret Labs AB
       Copyright © 1995-2011 by Fredrik Lundh and contributors

   Pillow is the friendly PIL fork. It is

       Copyright © 2010 by Jeffrey A. Clark and contributors

   Like PIL, Pillow is licensed under the open source MIT-CMU License:

   By obtaining, using, and/or copying this software and/or its associated
   documentation, you agree that you have read, understood, and will comply
   with the following terms and conditions:

   Permission to use, copy, modify and distribute this software and its
   documentation for any purpose and without fee is hereby granted,
   provided that the above copyright notice appears in all copies, and that
   both that copyright notice and this permission notice appear in supporting
   documentation, and that the name of Secret Labs AB or the author not be
   used in advertising or publicity pertaining to distribution of the software
   without specific, written prior permission.

   SECRET LABS AB AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS
   SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS.
   IN NO EVENT SHALL SECRET LABS AB OR THE AUTHOR BE LIABLE FOR ANY SPECIAL,
   INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
   LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
   OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
   PERFORMANCE OF THIS SOFTWARE.

   ```

## PyAV - [BSD 3-Clause "New" or "Revised" License](https://github.com/PyAV-Org/PyAV/blob/main/LICENSE.txt)

   ```

   Copyright retained by original committers. All rights reserved.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are met:
       * Redistributions of source code must retain the above copyright
         notice, this list of conditions and the following disclaimer.
       * Redistributions in binary form must reproduce the above copyright
         notice, this list of conditions and the following disclaimer in the
         documentation and/or other materials provided with the distribution.
       * Neither the name of the project nor the names of its contributors may be
         used to endorse or promote products derived from this software without
         specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
   AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
   IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
   DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY DIRECT,
   INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
   BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
   OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
   EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

   ```

## Pytorch_Retinaface - [MIT License](https://github.com/biubug6/Pytorch_Retinaface/blob/master/LICENSE.MIT)

   ```
   MIT License

   Copyright (c) 2019

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in all
   copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE.
   ```

## Sentencepiece - [Apache License 2.0](https://github.com/google/sentencepiece/blob/master/LICENSE)

   ```

                                    Apache License
                              Version 2.0, January 2004
                           http://www.apache.org/licenses/

      TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

      1. Definitions.

         "License" shall mean the terms and conditions for use, reproduction,
         and distribution as defined by Sections 1 through 9 of this document.

         "Licensor" shall mean the copyright owner or entity authorized by
         the copyright owner that is granting the License.

         "Legal Entity" shall mean the union of the acting entity and all
         other entities that control, are controlled by, or are under common
         control with that entity. For the purposes of this definition,
         "control" means (i) the power, direct or indirect, to cause the
         direction or management of such entity, whether by contract or
         otherwise, or (ii) ownership of fifty percent (50%) or more of the
         outstanding shares, or (iii) beneficial ownership of such entity.

         "You" (or "Your") shall mean an individual or Legal Entity
         exercising permissions granted by this License.

         "Source" form shall mean the preferred form for making modifications,
         including but not limited to software source code, documentation
         source, and configuration files.

         "Object" form shall mean any form resulting from mechanical
         transformation or translation of a Source form, including but
         not limited to compiled object code, generated documentation,
         and conversions to other media types.

         "Work" shall mean the work of authorship, whether in Source or
         Object form, made available under the License, as indicated by a
         copyright notice that is included in or attached to the work
         (an example is provided in the Appendix below).

         "Derivative Works" shall mean any work, whether in Source or Object
         form, that is based on (or derived from) the Work and for which the
         editorial revisions, annotations, elaborations, or other modifications
         represent, as a whole, an original work of authorship. For the purposes
         of this License, Derivative Works shall not include works that remain
         separable from, or merely link (or bind by name) to the interfaces of,
         the Work and Derivative Works thereof.

         "Contribution" shall mean any work of authorship, including
         the original version of the Work and any modifications or additions
         to that Work or Derivative Works thereof, that is intentionally
         submitted to Licensor for inclusion in the Work by the copyright owner
         or by an individual or Legal Entity authorized to submit on behalf of
         the copyright owner. For the purposes of this definition, "submitted"
         means any form of electronic, verbal, or written communication sent
         to the Licensor or its representatives, including but not limited to
         communication on electronic mailing lists, source code control systems,
         and issue tracking systems that are managed by, or on behalf of, the
         Licensor for the purpose of discussing and improving the Work, but
         excluding communication that is conspicuously marked or otherwise
         designated in writing by the copyright owner as "Not a Contribution."

         "Contributor" shall mean Licensor and any individual or Legal Entity
         on behalf of whom a Contribution has been received by Licensor and
         subsequently incorporated within the Work.

      2. Grant of Copyright License. Subject to the terms and conditions of
         this License, each Contributor hereby grants to You a perpetual,
         worldwide, non-exclusive, no-charge, royalty-free, irrevocable
         copyright license to reproduce, prepare Derivative Works of,
         publicly display, publicly perform, sublicense, and distribute the
         Work and such Derivative Works in Source or Object form.

      3. Grant of Patent License. Subject to the terms and conditions of
         this License, each Contributor hereby grants to You a perpetual,
         worldwide, non-exclusive, no-charge, royalty-free, irrevocable
         (except as stated in this section) patent license to make, have made,
         use, offer to sell, sell, import, and otherwise transfer the Work,
         where such license applies only to those patent claims licensable
         by such Contributor that are necessarily infringed by their
         Contribution(s) alone or by combination of their Contribution(s)
         with the Work to which such Contribution(s) was submitted. If You
         institute patent litigation against any entity (including a
         cross-claim or counterclaim in a lawsuit) alleging that the Work
         or a Contribution incorporated within the Work constitutes direct
         or contributory patent infringement, then any patent licenses
         granted to You under this License for that Work shall terminate
         as of the date such litigation is filed.

      4. Redistribution. You may reproduce and distribute copies of the
         Work or Derivative Works thereof in any medium, with or without
         modifications, and in Source or Object form, provided that You
         meet the following conditions:

         (a) You must give any other recipients of the Work or
             Derivative Works a copy of this License; and

         (b) You must cause any modified files to carry prominent notices
             stating that You changed the files; and

         (c) You must retain, in the Source form of any Derivative Works
             that You distribute, all copyright, patent, trademark, and
             attribution notices from the Source form of the Work,
             excluding those notices that do not pertain to any part of
             the Derivative Works; and

         (d) If the Work includes a "NOTICE" text file as part of its
             distribution, then any Derivative Works that You distribute must
             include a readable copy of the attribution notices contained
             within such NOTICE file, excluding those notices that do not
             pertain to any part of the Derivative Works, in at least one
             of the following places: within a NOTICE text file distributed
             as part of the Derivative Works; within the Source form or
             documentation, if provided along with the Derivative Works; or,
             within a display generated by the Derivative Works, if and
             wherever such third-party notices normally appear. The contents
             of the NOTICE file are for informational purposes only and
             do not modify the License. You may add Your own attribution
             notices within Derivative Works that You distribute, alongside
             or as an addendum to the NOTICE text from the Work, provided
             that such additional attribution notices cannot be construed
             as modifying the License.

         You may add Your own copyright statement to Your modifications and
         may provide additional or different license terms and conditions
         for use, reproduction, or distribution of Your modifications, or
         for any such Derivative Works as a whole, provided Your use,
         reproduction, and distribution of the Work otherwise complies with
         the conditions stated in this License.

      5. Submission of Contributions. Unless You explicitly state otherwise,
         any Contribution intentionally submitted for inclusion in the Work
         by You to the Licensor shall be under the terms and conditions of
         this License, without any additional terms or conditions.
         Notwithstanding the above, nothing herein shall supersede or modify
         the terms of any separate license agreement you may have executed
         with Licensor regarding such Contributions.

      6. Trademarks. This License does not grant permission to use the trade
         names, trademarks, service marks, or product names of the Licensor,
         except as required for reasonable and customary use in describing the
         origin of the Work and reproducing the content of the NOTICE file.

      7. Disclaimer of Warranty. Unless required by applicable law or
         agreed to in writing, Licensor provides the Work (and each
         Contributor provides its Contributions) on an "AS IS" BASIS,
         WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
         implied, including, without limitation, any warranties or conditions
         of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
         PARTICULAR PURPOSE. You are solely responsible for determining the
         appropriateness of using or redistributing the Work and assume any
         risks associated with Your exercise of permissions under this License.

      8. Limitation of Liability. In no event and under no legal theory,
         whether in tort (including negligence), contract, or otherwise,
         unless required by applicable law (such as deliberate and grossly
         negligent acts) or agreed to in writing, shall any Contributor be
         liable to You for damages, including any direct, indirect, special,
         incidental, or consequential damages of any character arising as a
         result of this License or out of the use or inability to use the
         Work (including but not limited to damages for loss of goodwill,
         work stoppage, computer failure or malfunction, or any and all
         other commercial damages or losses), even if such Contributor
         has been advised of the possibility of such damages.

      9. Accepting Warranty or Additional Liability. While redistributing
         the Work or Derivative Works thereof, You may choose to offer,
         and charge a fee for, acceptance of support, warranty, indemnity,
         or other liability obligations and/or rights consistent with this
         License. However, in accepting such obligations, You may act only
         on Your own behalf and on Your sole responsibility, not on behalf
         of any other Contributor, and only if You agree to indemnify,
         defend, and hold each Contributor harmless for any liability
         incurred by, or claims asserted against, such Contributor by reason
         of your accepting any such warranty or additional liability.

      END OF TERMS AND CONDITIONS

      APPENDIX: How to apply the Apache License to your work.

         To apply the Apache License to your work, attach the following
         boilerplate notice, with the fields enclosed by brackets "[]"
         replaced with your own identifying information. (Don't include
         the brackets!)  The text should be enclosed in the appropriate
         comment syntax for the file format. We also recommend that a
         file or class name and description of purpose be included on the
         same "printed page" as the copyright notice for easier
         identification within third-party archives.

      Copyright [yyyy] [name of copyright owner]

      Licensed under the Apache License, Version 2.0 (the "License");
      you may not use this file except in compliance with the License.
      You may obtain a copy of the License at

          http://www.apache.org/licenses/LICENSE-2.0

      Unless required by applicable law or agreed to in writing, software
      distributed under the License is distributed on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
      See the License for the specific language governing permissions and
      limitations under the License.

   ```

## Termcolor - [MIT License](https://github.com/termcolor/termcolor/blob/main/COPYING.txt)

   ```
   Copyright (c) 2008-2011 Volvox Development Team

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
   ```

## Transformers [Apache License 2.0](https://github.com/huggingface/transformers/blob/main/LICENSE)

   ```

   Copyright 2018- The Hugging Face team. All rights reserved.

                                    Apache License
                              Version 2.0, January 2004
                           http://www.apache.org/licenses/

      TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

      1. Definitions.

         "License" shall mean the terms and conditions for use, reproduction,
         and distribution as defined by Sections 1 through 9 of this document.

         "Licensor" shall mean the copyright owner or entity authorized by
         the copyright owner that is granting the License.

         "Legal Entity" shall mean the union of the acting entity and all
         other entities that control, are controlled by, or are under common
         control with that entity. For the purposes of this definition,
         "control" means (i) the power, direct or indirect, to cause the
         direction or management of such entity, whether by contract or
         otherwise, or (ii) ownership of fifty percent (50%) or more of the
         outstanding shares, or (iii) beneficial ownership of such entity.

         "You" (or "Your") shall mean an individual or Legal Entity
         exercising permissions granted by this License.

         "Source" form shall mean the preferred form for making modifications,
         including but not limited to software source code, documentation
         source, and configuration files.

         "Object" form shall mean any form resulting from mechanical
         transformation or translation of a Source form, including but
         not limited to compiled object code, generated documentation,
         and conversions to other media types.

         "Work" shall mean the work of authorship, whether in Source or
         Object form, made available under the License, as indicated by a
         copyright notice that is included in or attached to the work
         (an example is provided in the Appendix below).

         "Derivative Works" shall mean any work, whether in Source or Object
         form, that is based on (or derived from) the Work and for which the
         editorial revisions, annotations, elaborations, or other modifications
         represent, as a whole, an original work of authorship. For the purposes
         of this License, Derivative Works shall not include works that remain
         separable from, or merely link (or bind by name) to the interfaces of,
         the Work and Derivative Works thereof.

         "Contribution" shall mean any work of authorship, including
         the original version of the Work and any modifications or additions
         to that Work or Derivative Works thereof, that is intentionally
         submitted to Licensor for inclusion in the Work by the copyright owner
         or by an individual or Legal Entity authorized to submit on behalf of
         the copyright owner. For the purposes of this definition, "submitted"
         means any form of electronic, verbal, or written communication sent
         to the Licensor or its representatives, including but not limited to
         communication on electronic mailing lists, source code control systems,
         and issue tracking systems that are managed by, or on behalf of, the
         Licensor for the purpose of discussing and improving the Work, but
         excluding communication that is conspicuously marked or otherwise
         designated in writing by the copyright owner as "Not a Contribution."

         "Contributor" shall mean Licensor and any individual or Legal Entity
         on behalf of whom a Contribution has been received by Licensor and
         subsequently incorporated within the Work.

      2. Grant of Copyright License. Subject to the terms and conditions of
         this License, each Contributor hereby grants to You a perpetual,
         worldwide, non-exclusive, no-charge, royalty-free, irrevocable
         copyright license to reproduce, prepare Derivative Works of,
         publicly display, publicly perform, sublicense, and distribute the
         Work and such Derivative Works in Source or Object form.

      3. Grant of Patent License. Subject to the terms and conditions of
         this License, each Contributor hereby grants to You a perpetual,
         worldwide, non-exclusive, no-charge, royalty-free, irrevocable
         (except as stated in this section) patent license to make, have made,
         use, offer to sell, sell, import, and otherwise transfer the Work,
         where such license applies only to those patent claims licensable
         by such Contributor that are necessarily infringed by their
         Contribution(s) alone or by combination of their Contribution(s)
         with the Work to which such Contribution(s) was submitted. If You
         institute patent litigation against any entity (including a
         cross-claim or counterclaim in a lawsuit) alleging that the Work
         or a Contribution incorporated within the Work constitutes direct
         or contributory patent infringement, then any patent licenses
         granted to You under this License for that Work shall terminate
         as of the date such litigation is filed.

      4. Redistribution. You may reproduce and distribute copies of the
         Work or Derivative Works thereof in any medium, with or without
         modifications, and in Source or Object form, provided that You
         meet the following conditions:

         (a) You must give any other recipients of the Work or
             Derivative Works a copy of this License; and

         (b) You must cause any modified files to carry prominent notices
             stating that You changed the files; and

         (c) You must retain, in the Source form of any Derivative Works
             that You distribute, all copyright, patent, trademark, and
             attribution notices from the Source form of the Work,
             excluding those notices that do not pertain to any part of
             the Derivative Works; and

         (d) If the Work includes a "NOTICE" text file as part of its
             distribution, then any Derivative Works that You distribute must
             include a readable copy of the attribution notices contained
             within such NOTICE file, excluding those notices that do not
             pertain to any part of the Derivative Works, in at least one
             of the following places: within a NOTICE text file distributed
             as part of the Derivative Works; within the Source form or
             documentation, if provided along with the Derivative Works; or,
             within a display generated by the Derivative Works, if and
             wherever such third-party notices normally appear. The contents
             of the NOTICE file are for informational purposes only and
             do not modify the License. You may add Your own attribution
             notices within Derivative Works that You distribute, alongside
             or as an addendum to the NOTICE text from the Work, provided
             that such additional attribution notices cannot be construed
             as modifying the License.

         You may add Your own copyright statement to Your modifications and
         may provide additional or different license terms and conditions
         for use, reproduction, or distribution of Your modifications, or
         for any such Derivative Works as a whole, provided Your use,
         reproduction, and distribution of the Work otherwise complies with
         the conditions stated in this License.

      5. Submission of Contributions. Unless You explicitly state otherwise,
         any Contribution intentionally submitted for inclusion in the Work
         by You to the Licensor shall be under the terms and conditions of
         this License, without any additional terms or conditions.
         Notwithstanding the above, nothing herein shall supersede or modify
         the terms of any separate license agreement you may have executed
         with Licensor regarding such Contributions.

      6. Trademarks. This License does not grant permission to use the trade
         names, trademarks, service marks, or product names of the Licensor,
         except as required for reasonable and customary use in describing the
         origin of the Work and reproducing the content of the NOTICE file.

      7. Disclaimer of Warranty. Unless required by applicable law or
         agreed to in writing, Licensor provides the Work (and each
         Contributor provides its Contributions) on an "AS IS" BASIS,
         WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
         implied, including, without limitation, any warranties or conditions
         of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
         PARTICULAR PURPOSE. You are solely responsible for determining the
         appropriateness of using or redistributing the Work and assume any
         risks associated with Your exercise of permissions under this License.

      8. Limitation of Liability. In no event and under no legal theory,
         whether in tort (including negligence), contract, or otherwise,
         unless required by applicable law (such as deliberate and grossly
         negligent acts) or agreed to in writing, shall any Contributor be
         liable to You for damages, including any direct, indirect, special,
         incidental, or consequential damages of any character arising as a
         result of this License or out of the use or inability to use the
         Work (including but not limited to damages for loss of goodwill,
         work stoppage, computer failure or malfunction, or any and all
         other commercial damages or losses), even if such Contributor
         has been advised of the possibility of such damages.

      9. Accepting Warranty or Additional Liability. While redistributing
         the Work or Derivative Works thereof, You may choose to offer,
         and charge a fee for, acceptance of support, warranty, indemnity,
         or other liability obligations and/or rights consistent with this
         License. However, in accepting such obligations, You may act only
         on Your own behalf and on Your sole responsibility, not on behalf
         of any other Contributor, and only if You agree to indemnify,
         defend, and hold each Contributor harmless for any liability
         incurred by, or claims asserted against, such Contributor by reason
         of your accepting any such warranty or additional liability.

      END OF TERMS AND CONDITIONS

      APPENDIX: How to apply the Apache License to your work.

         To apply the Apache License to your work, attach the following
         boilerplate notice, with the fields enclosed by brackets "[]"
         replaced with your own identifying information. (Don't include
         the brackets!)  The text should be enclosed in the appropriate
         comment syntax for the file format. We also recommend that a
         file or class name and description of purpose be included on the
         same "printed page" as the copyright notice for easier
         identification within third-party archives.

      Copyright [yyyy] [name of copyright owner]

      Licensed under the Apache License, Version 2.0 (the "License");
      you may not use this file except in compliance with the License.
      You may obtain a copy of the License at

          http://www.apache.org/licenses/LICENSE-2.0

      Unless required by applicable law or agreed to in writing, software
      distributed under the License is distributed on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
      See the License for the specific language governing permissions and
      limitations under the License.

   ```

## MoGe [MIT License](https://github.com/microsoft/MoGe/blob/main/LICENSE)

   ```
    MIT License

    Copyright (c) Microsoft Corporation.

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE
   ```

## Warp [Apache License 2.0](https://github.com/NVIDIA/warp/blob/main/LICENSE.md)

   ```
   
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   ```

## Args - [MIT License](https://github.com/Taywee/args/blob/master/LICENSE)

   ```
   
   Copyright (c) 2016-2024 Taylor C. Richberger <taylor@axfive.net> and Pavel Belikov


   Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

   ```

## CUDA CMake GitHub Actions - [MIT License](https://github.com/ptheywood/cuda-cmake-github-actions/blob/master/LICENSE)

   ```
   
   MIT License

   Copyright (c) 2021 Peter Heywood

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in all
   copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE.

   ```

## Dear ImGui - [MIT License](https://github.com/ocornut/imgui/blob/master/LICENSE.txt)

   ```
   
   The MIT License (MIT)

   Copyright (c) 2014-2024 Omar Cornut

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in all
   copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE.

   ```

## Filesystem - BSD 3-Clause License

   ```
   
   Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>,
                 2021-2023 Thomas Müller

   All rights reserved.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are met:

   1. Redistributions of source code must retain the above copyright notice, this
      list of conditions and the following disclaimer.

   2. Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.

   3. Neither the name of the copyright holder nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
   ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
   WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
   DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
   FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
   DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
   SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
   CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
   OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

   You are under no obligation whatsoever to provide any bug fixes, patches, or
   upgrades to the features, functionality or performance of the source code
   ("Enhancements") to anyone; however, if you choose to make your Enhancements
   available either publicly, or directly to the author of this software, without
   imposing a separate written license agreement for such Enhancements, then you
   hereby grant the following license: a non-exclusive, royalty-free perpetual
   license to install, use, modify, prepare derivative works, incorporate into
   other computer software, distribute, and sublicense such enhancements or
   derivative works thereof, in binary and source code form.

   ```

## fmt - [MIT License](https://github.com/fmtlib/fmt/blob/master/LICENSE)

   ```
   
   Copyright (c) 2012 - present, Victor Zverovich and {fmt} contributors

   Permission is hereby granted, free of charge, to any person obtaining
   a copy of this software and associated documentation files (the
   "Software"), to deal in the Software without restriction, including
   without limitation the rights to use, copy, modify, merge, publish,
   distribute, sublicense, and/or sell copies of the Software, and to
   permit persons to whom the Software is furnished to do so, subject to
   the following conditions:

   The above copyright notice and this permission notice shall be
   included in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
   EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
   MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
   NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
   LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
   OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
   WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

   --- Optional exception to the license ---

   As an exception, if, as a result of your compiling your source code, portions
   of this Software are embedded into a machine-executable object form of such
   source code, you may redistribute such embedded portions in such object form
   without including the above copyright and permission notices.

   ```

## GLFW - [zlib/libpng License](https://github.com/glfw/glfw/blob/master/LICENSE.md)

   ```
   
   Copyright (c) 2002-2006 Marcus Geelnard
   Copyright (c) 2006-2019 Camilla Löwy

   This software is provided 'as-is', without any express or implied
   warranty. In no event will the authors be held liable for any damages
   arising from the use of this software.

   Permission is granted to anyone to use this software for any purpose,
   including commercial applications, and to alter it and redistribute it
   freely, subject to the following restrictions:

   1. The origin of this software must not be misrepresented; you must not
      claim that you wrote the original software. If you use this software
      in a product, an acknowledgment in the product documentation would
      be appreciated but is not required.

   2. Altered source versions must be plainly marked as such, and must not
      be misrepresented as being the original software.

   3. This notice may not be removed or altered from any source
      distribution.

   ```

## JSON for Modern C++ - [MIT License](https://github.com/nlohmann/json/tree/develop/LICENSES)

   ```
   
   MIT License 

   Copyright (c) 2013-2021 Niels Lohmann

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in all
   copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE.

   ```


## PlayNE Equivalence - MIT License

   ```
   
   MIT License

   Copyright (c) 2018 - Daniel Peter Playne

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in all
   copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE.

   ```

## pybind11_json - [BSD 3-Clause License](https://github.com/pybind/pybind11_json/blob/master/LICENSE)

   ```
   
   BSD 3-Clause License

   Copyright (c) 2019,
   All rights reserved.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are met:

   * Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

   * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

   * Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
   AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
   IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
   DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
   FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
   DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
   SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
   CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
   OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

   ```


## stb_image - [MIT OR Public Domain](https://github.com/nothings/stb/blob/master/LICENSE)

   ```
   
   This software is available under 2 licenses -- choose whichever you prefer.
   ------------------------------------------------------------------------------
   ALTERNATIVE A - MIT License
   Copyright (c) 2017 Sean Barrett
   Permission is hereby granted, free of charge, to any person obtaining a copy of
   this software and associated documentation files (the "Software"), to deal in
   the Software without restriction, including without limitation the rights to
   use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
   of the Software, and to permit persons to whom the Software is furnished to do
   so, subject to the following conditions:
   The above copyright notice and this permission notice shall be included in all
   copies or substantial portions of the Software.
   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE.
   ------------------------------------------------------------------------------
   ALTERNATIVE B - Public Domain (www.unlicense.org)
   This is free and unencumbered software released into the public domain.
   Anyone is free to copy, modify, publish, use, compile, sell, or distribute this
   software, either in source code form or as a compiled binary, for any purpose,
   commercial or non-commercial, and by any means.
   In jurisdictions that recognize copyright laws, the author or authors of this
   software dedicate any and all copyright interest in the software to the public
   domain. We make this dedication for the benefit of the public at large and to
   the detriment of our heirs and successors. We intend this dedication to be an
   overt act of relinquishment in perpetuity of all present and future rights to
   this software under copyright law.
   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
   ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
   WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

   ```

## ImGuizmo - [MIT license](https://github.com/CedricGuillemet/ImGuizmo/blob/master/LICENSE)
   ```
   The MIT License (MIT)

   Copyright (c) 2016 Cedric Guillemet

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in all
   copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE.
   ```

## PCG - [Apache-2.0 License]()
   ```
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "{}"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright {yyyy} {name of copyright owner}

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

   ```

## OpenXR-SDK - [Apache License 2.0](https://github.com/KhronosGroup/OpenXR-SDK/blob/main/LICENSE)
   ```
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

   ```

## pybind11 - [BSD 3-Clause License](https://github.com/Tom94/pybind11/blob/master/LICENSE)
   ```
   Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>, All rights reserved.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are met:

   1. Redistributions of source code must retain the above copyright notice, this
      list of conditions and the following disclaimer.

   2. Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.

   3. Neither the name of the copyright holder nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
   ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
   WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
   DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
   FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
   DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
   SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
   CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
   OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

   Please also refer to the file .github/CONTRIBUTING.md, which clarifies licensing of
   external contributions to this project including patches, pull requests, etc.

   ```

## tinylogger - [BSD-3-Clause license](https://github.com/Tom94/tinylogger/blob/master/LICENSE.md)
   ```
   Copyright (c) 2018-2024, Thomas Müller <contact@tom94.net>
   All rights reserved.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are met:

   * Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

   * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

   * Neither the name of the copyright holder, the project, nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
   AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
   IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
   DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
   FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
   DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
   SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
   CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
   OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

   ```

## ZLIB DATA COMPRESSION LIBRARY - [zlib License](https://github.com/Tom94/zlib)
   ```
   (C) 1995-2017 Jean-loup Gailly and Mark Adler

   This software is provided 'as-is', without any express or implied
   warranty.  In no event will the authors be held liable for any damages
   arising from the use of this software.

   Permission is granted to anyone to use this software for any purpose,
   including commercial applications, and to alter it and redistribute it
   freely, subject to the following restrictions:

   1. The origin of this software must not be misrepresented; you must not
      claim that you wrote the original software. If you use this software
      in a product, an acknowledgment in the product documentation would be
      appreciated but is not required.
   2. Altered source versions must be plainly marked as such, and must not be
      misrepresented as being the original software.
   3. This notice may not be removed or altered from any source distribution.

   Jean-loup Gailly        Mark Adler
   jloup@gzip.org          madler@alumni.caltech.edu

   If you use the zlib library in a product, we would appreciate *not* receiving
   lengthy legal documents to sign.  The sources are provided for free but without
   warranty of any kind.  The library has been entirely written by Jean-loup
   Gailly and Mark Adler; it does not include third-party code.

   If you redistribute modified sources, we would appreciate that you include in
   the file ChangeLog history information documenting your changes.  Please read
   the FAQ for more information on the distribution of modified source versions.

   ```

## GL3W - [Unlicense License](https://github.com/skeeto/opengl-demo/blob/master/UNLICENSE)
   ```
   This is free and unencumbered software released into the public domain.

   Anyone is free to copy, modify, publish, use, compile, sell, or
   distribute this software, either in source code form or as a compiled
   binary, for any purpose, commercial or non-commercial, and by any
   means.

   In jurisdictions that recognize copyright laws, the author or authors
   of this software dedicate any and all copyright interest in the
   software to the public domain. We make this dedication for the benefit
   of the public at large and to the detriment of our heirs and
   successors. We intend this dedication to be an overt act of
   relinquishment in perpetuity of all present and future rights to this
   software under copyright law.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
   EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
   MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
   IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
   OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
   ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
   OTHER DEALINGS IN THE SOFTWARE.

   For more information, please refer to <http://unlicense.org/>
   ```


## DLSS - [NVIDIA RTX SDKs LICENSE](https://github.com/NVIDIA/DLSS/blob/main/LICENSE.txt)
   ```
   NVIDIA RTX SDKs LICENSE

   This license is a legal agreement between you and NVIDIA Corporation ("NVIDIA") and governs the use of the NVIDIA RTX software development kits, including the DLSS SDK, NGX SDK, RTXGI SDK, RTXDI SDK, RTX Video SDK, RTX Dynamic Vibrance SDK and/or NRD SDK, if and when made available to you under this license (in each case, the “SDK”).
   This license can be accepted only by an adult of legal age of majority in the country in which the SDK is used. If you are under the legal age of majority, you must ask your parent or legal guardian to consent to this license. If you are entering this license on behalf of a company or other legal entity, you represent that you have legal authority and “you” will mean the entity you represent. 
   By using the SDK, you affirm that you have reached the legal age of majority, you accept the terms of this license, and you take legal and financial responsibility for the actions of your permitted users. 

   You agree to use the SDK only for purposes that are permitted by (a) this license, and (b) any applicable law, regulation or generally accepted practices or guidelines in the relevant jurisdictions.

   1. LICENSE. Subject to the terms of this license and the terms in the supplement attached, NVIDIA hereby grants you a non-exclusive, non-transferable license, without the right to sublicense (except as expressly provided in this license) to: 
   a. Install and use the SDK,
   b. Modify and create derivative works of sample source code delivered in the SDK, and
   c. Distribute any software and materials within the SDK, other than developer tools provided for your internal use, as incorporated in object code format into a software application subject to the distribution requirements indicated in this license.

   2. DISTRIBUTION REQUIREMENTS. These are the distribution requirements for you to exercise the grants above:
   a.	An application must have material additional functionality, beyond the included portions of the SDK. 
   b.	The following notice shall be included in modifications and derivative works of source code distributed: “This software contains source code provided by NVIDIA Corporation.” 
   c.	You agree to distribute the SDK subject to the terms at least as protective as the terms of this license, including (without limitation) terms relating to the license grant, license restrictions and protection of NVIDIA’s intellectual property rights. Additionally, you agree that you will protect the privacy, security and legal rights of your application users. 
   d.	You agree to notify NVIDIA in writing of any known or suspected distribution or use of the SDK not in compliance with the requirements of this license, and to enforce the terms of your agreements with respect to the distributed portions of the SDK. 

   3. AUTHORIZED USERS. You may allow employees and contractors of your entity or of your subsidiary(ies) to access and use the SDK from your secure network to perform work on your behalf. If you are an academic institution you may allow users enrolled or employed by the academic institution to access and use the SDK from your secure network. You are responsible for the compliance with the terms of this license by your authorized users. 

   4. LIMITATIONS. Your license to use the SDK is restricted as follows:
   a.	You may not reverse engineer, decompile or disassemble, or remove copyright or other proprietary notices from any portion of the SDK or copies of the SDK. 
   b.	Except as expressly provided in this license, you may not copy, sell, rent, sublicense, transfer, distribute, modify, or create derivative works of any portion of the SDK. For clarity, you may not distribute or sublicense the SDK as a stand-alone product.
   c.	 Unless you have an agreement with NVIDIA for this purpose, you may not indicate that an application created with the SDK is sponsored or endorsed by NVIDIA.
   d.	 You may not bypass, disable, or circumvent any technical limitation, encryption, security, digital rights management or authentication mechanism in the SDK.  
   e.	You may not use the SDK in any manner that would cause it to become subject to an open source software license. As examples, licenses that require as a condition of use, modification, and/or distribution that the SDK be: (i) disclosed or distributed in source code form; (ii) licensed for the purpose of making derivative works; or (iii) redistributable at no charge.
   f.	 Unless you have an agreement with NVIDIA for this purpose, you may not use the SDK with any system or application where the use or failure of the system or application can reasonably be expected to threaten or result in personal injury, death, or catastrophic loss. Examples include use in avionics, navigation, military, medical, life support or other life critical applications. NVIDIA does not design, test or manufacture the SDK for these critical uses and NVIDIA shall not be liable to you or any third party, in whole or in part, for any claims or damages arising from such uses. 
   g.	You agree to defend, indemnify and hold harmless NVIDIA and its affiliates, and their respective employees, contractors, agents, officers and directors, from and against any and all claims, damages, obligations, losses, liabilities, costs or debt, fines, restitutions and expenses (including but not limited to attorney’s fees and costs incident to establishing the right of indemnification) arising out of or related to your use of the SDK outside of the scope of this license, or not in compliance with its terms. 

   5. UPDATES. NVIDIA may, at its option, make available patches, workarounds or other updates to this SDK. Unless the updates are provided with their separate governing terms, they are deemed part of the SDK licensed to you as provided in this license. Further, NVIDIA may, at its option, automatically update the SDK or other software in the system, except for those updates that you may opt-out via the SDK API. You agree that the form and content of the SDK that NVIDIA provides may change without prior notice to you. While NVIDIA generally maintains compatibility between versions, NVIDIA may in some cases make changes that introduce incompatibilities in future versions of the SDK.

   6. PRE-RELEASE VERSIONS. SDK versions identified as alpha, beta, preview, early access or otherwise as pre-release may not be fully functional, may contain errors or design flaws, and may have reduced or different security, privacy, availability, and reliability standards relative to commercial versions of NVIDIA software and materials. You may use a pre-release SDK version at your own risk, understanding that these versions are not intended for use in production or business-critical systems. NVIDIA may choose not to make available a commercial version of any pre-release SDK. NVIDIA may also choose to abandon development and terminate the availability of a pre-release SDK at any time without liability.

   7. THIRD-PARTY COMPONENTS. The SDK may include third-party components with separate legal notices or terms as may be described in proprietary notices accompanying the SDK. If and to the extent there is a conflict between the terms in this license and the third-party license terms, the third-party terms control only to the extent necessary to resolve the conflict. 

   8. OWNERSHIP. 

   8.1 NVIDIA reserves all rights, title and interest in and to the SDK not expressly granted to you under this license. NVIDIA and its suppliers hold all rights, title and interest in and to the SDK, including their respective intellectual property rights. The SDK is copyrighted and protected by the laws of the United States and other countries, and international treaty provisions.

   8.2 Subject to the rights of NVIDIA and its suppliers in the SDK, you hold all rights, title and interest in and to your applications and your derivative works of the sample source code delivered in the SDK including their respective intellectual property rights. 
   
   9. FEEDBACK. You may, but are not obligated to, provide Feedback to NVIDIA. “Feedback” means all suggestions, fixes, modifications, feature requests or other feedback regarding the SDK. Feedback, even if designated as confidential by you, shall not create any confidentiality obligation for NVIDIA. If you provide Feedback, you hereby grant NVIDIA, its affiliates and its designees a non-exclusive, perpetual, irrevocable, sublicensable, worldwide, royalty-free, fully paid-up and transferable license, under your intellectual property rights, to publicly perform, publicly display, reproduce, use, make, have made, sell, offer for sale, distribute (through multiple tiers of distribution), import, create derivative works of and otherwise commercialize and exploit the Feedback at NVIDIA’s discretion. You will not give Feedback (i) that you have reason to believe is subject to any restriction that impairs the exercise of the grant stated in this section, such as third-party intellectual property rights or (ii) subject to license terms which seek to require any product incorporating or developed using such Feedback, or other intellectual property of NVIDIA or its affiliates, to be licensed to or otherwise shared with any third party. 

   10. NO WARRANTIES. THE SDK IS PROVIDED AS-IS. TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW NVIDIA AND ITS AFFILIATES EXPRESSLY DISCLAIM ALL WARRANTIES OF ANY KIND OR NATURE, WHETHER EXPRESS, IMPLIED OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, WARRANTIES OF MERCHANTABILITY, NON-INFRINGEMENT, FITNESS FOR A PARTICULAR PURPOSE, USAGE OF TRADE AND COURSE OF DEALING. NVIDIA DOES NOT WARRANT THAT THE SDK WILL MEET YOUR REQUIREMENTS OR THAT THE OPERATION THEREOF WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ALL ERRORS WILL BE CORRECTED. 

   11. LIMITATIONS OF LIABILITY. TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW NVIDIA AND ITS AFFILIATES SHALL NOT BE LIABLE FOR ANY (I) SPECIAL, INCIDENTAL, PUNITIVE OR CONSEQUENTIAL DAMAGES, OR FOR DAMAGES FOR  (A) ANY LOST PROFITS, PROJECT DELAYS, LOSS OF USE, LOSS OF DATA OR LOSS OF GOODWILL, OR (B) THE COSTS OF PROCURING SUBSTITUTE PRODUCTS, ARISING OUT OF OR IN CONNECTION WITH THIS LICENSE OR THE USE OR PERFORMANCE OF THE SDK, WHETHER SUCH LIABILITY ARISES FROM ANY CLAIM BASED UPON BREACH OF CONTRACT, BREACH OF WARRANTY, TORT (INCLUDING NEGLIGENCE), PRODUCT LIABILITY OR ANY OTHER CAUSE OF ACTION OR THEORY OF LIABILITY, EVEN IF NVIDIA HAS PREVIOUSLY BEEN ADVISED OF, OR COULD REASONABLY HAVE FORESEEN, THE POSSIBILITY OF SUCH DAMAGES. IN NO EVENT WILL NVIDIA’S AND ITS AFFILIATES TOTAL CUMULATIVE LIABILITY UNDER OR ARISING OUT OF THIS LICENSE EXCEED US$10.00. THE NATURE OF THE LIABILITY OR THE NUMBER OF CLAIMS OR SUITS SHALL NOT ENLARGE OR EXTEND THIS LIMIT. 

   12. TERMINATION. Your rights under this license will terminate automatically without notice from NVIDIA if you fail to comply with any term and condition of this license or if you commence or participate in any legal proceeding against NVIDIA with respect to the SDK. NVIDIA may terminate this license with advance written notice to you, if NVIDIA decides to no longer provide the SDK in a country or, in NVIDIA’s sole discretion, the continued use of it is no longer commercially viable. Upon any termination of this license, you agree to promptly discontinue use of the SDK and destroy all copies in your possession or control. Your prior distributions in accordance with this license are not affected by the termination of this license. All provisions of this license will survive termination, except for the license granted to you. 

   13. APPLICABLE LAW. This license will be governed in all respects by the laws of the United States and of the State of Delaware, without regard to the conflicts of laws principles. The United Nations Convention on Contracts for the International Sale of Goods is specifically disclaimed. You agree to all terms of this license in the English language. The state or federal courts residing in Santa Clara County, California shall have exclusive jurisdiction over any dispute or claim arising out of this license. Notwithstanding this, you agree that NVIDIA shall still be allowed to apply for injunctive remedies or urgent legal relief in any jurisdiction. 

   14. NO ASSIGNMENT. This license and your rights and obligations thereunder may not be assigned by you by any means or operation of law without NVIDIA’s permission. Any attempted assignment not approved by NVIDIA in writing shall be void and of no effect. NVIDIA may assign, delegate or transfer this license and its rights and obligations, and if to a non-affiliate you will be notified.
   
   15. EXPORT. The SDK is subject to United States export laws and regulations. You agree to comply with all applicable U.S. and international export laws, including the Export Administration Regulations (EAR) administered by the U.S. Department of Commerce and economic sanctions administered by the U.S. Department of Treasury’s Office of Foreign Assets Control (OFAC). These laws include restrictions on destinations, end-users and end-use. By accepting this license, you confirm that you are not currently residing in a country or region currently embargoed by the U.S. and that you are not otherwise prohibited from receiving the SDK.

   16. GOVERNMENT USE. The SDK, documentation and technology (“Protected Items”) are “Commercial products” as this term is defined at 48 C.F.R. 2.101, consisting of “commercial computer software” and “commercial computer software documentation” as such terms are used in, respectively, 48 C.F.R. 12.212 and 48 C.F.R. 227.7202 & 252.227-7014(a)(1). Before any Protected Items are supplied to the U.S. Government, you will (i) inform the U.S. Government in writing that the Protected Items are and must be treated as commercial computer software and commercial computer software documentation developed at private expense; (ii) inform the U.S. Government that the Protected Items are provided subject to the terms of the Agreement; and (iii) mark the Protected Items as commercial computer software and commercial computer software documentation developed at private expense. In no event will you permit the U.S. Government to acquire rights in Protected Items beyond those specified in 48 C.F.R. 52.227-19(b)(1)-(2) or 252.227-7013(c) except as expressly approved by NVIDIA in writing.

   17. NOTICES. You agree that any notices that NVIDIA sends you electronically, such as via email, will satisfy any legal communication requirements. Please direct your legal notices or other correspondence to NVIDIA Corporation, 2788 San Tomas Expressway, Santa Clara, California 95051, United States of America, Attention: Legal Department.

   18. ENTIRE AGREEMENT. This license is the final, complete and exclusive agreement between the parties relating to the subject matter of this license and supersedes all prior or contemporaneous understandings and agreements relating to this subject matter, whether oral or written. If any court of competent jurisdiction determines that any provision of this license is illegal, invalid or unenforceable, the remaining provisions will remain in full force and effect. Any amendment or waiver under this license shall be in writing and signed by representatives of both parties.

   19. LICENSING. If the distribution terms in this license are not suitable for your organization, or for any questions regarding this license, please contact NVIDIA at nvidia-rtx-license-questions@nvidia.com.
   (v. March 14, 2024)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    

   NVIDIA RTX SUPPLEMENT TO SOFTWARE LICENSE AGREEMENT FOR NVIDIA SOFTWARE DEVELOPMENT KITS
   The terms in this supplement govern your use of the NVIDIA RTX SDKs, including the DLSS SDK, NGX SDK, RTXGI SDK, RTXDI SDK, RTX Video SDK, RTX Dynamic Vibrance SDK and/or NRD SDK, if and when made available to you (in each case, the “SDK”) under the terms of your license agreement (“Agreement”) as modified by this supplement. Capitalized terms used but not defined below have the meaning assigned to them in the Agreement.
   This supplement is an exhibit to the Agreement and is incorporated as an integral part of the Agreement. In the event of conflict between the terms in this supplement and the terms in the Agreement, the terms in this supplement govern.  

   1. Interoperability. Your applications that incorporate, or are based on, the SDK must be fully interoperable with compatible GPU hardware products designed by NVIDIA or its affiliates. Further, the DLSS SDK, NGX SDK and RTX Dynamic Vibrance SDK are licensed for you to develop applications only for their use in systems with NVIDIA GPUs.

   2. Game License. You may, but are not obligated to, provide your game or related content (“Game Content”) to NVIDIA. If you provide Game Content, you hereby grant NVIDIA, its affiliates and its designees a non-exclusive, perpetual, irrevocable, worldwide, royalty-free, fully paid-up license, to use the Game Content to improve NVIDIA DLSS SDK and DLSS Model Training. 

   3. Limitations for the DLSS SDK, NGX SDK,RTX Video SDK and RTX Dynamic Vibrance SDK. Your applications that incorporate, or are based on, the DLSS SDK, NGX SDK, RTX Video SDK or RTX Dynamic Vibrance SDK may be deployed in a cloud service that runs on systems that consume NVIDIA vGPU software, and any other cloud service use of such SDKs or their functionality is outside of the scope of the Agreement. For the purpose of this section, cloud services include application service providers or service bureaus, operators of hosted/virtual system environments, or hosting, time sharing or providing any other type of service to others. 

   4. Notification for the DLSS SDK, NGX SDK and RTX Dynamic Vibrance SDK. You are required to notify NVIDIA prior to commercial release of an application (including a plug-in to a commercial application) that incorporates, or is based on, the DLSS SDK, NGX SDK or RTX Dynamic Vibrance SDK. Please send notifications to: https://developer.nvidia.com/sw-notification and provide the following information in the email: company name, publisher and developer name, NVIDIA SDK used, application name, platform (i.e. PC, Linux), scheduled ship date, and weblink to product/video. 

   5. Audio and Video Encoders and Decoders. You acknowledge and agree that it is your sole responsibility to obtain any additional third-party licenses required to make, have made, use, have used, sell, import, and offer for sale your products or services that include or incorporate any third-party software and content relating to audio and/or video encoders and decoders from, including but not limited to, Microsoft, Thomson, Fraunhofer IIS, Sisvel S.p.A., MPEG-LA, and Coding Technologies. NVIDIA does not grant to you under this Agreement any necessary patent or other rights with respect to any audio and/or video encoders and decoders. 

   6. SDK Terms. 
   6.1 Over the Air Updates. By installing or using the SDK you agree that NVIDIA can make over-the-air updates of the SDK in systems that have the SDK installed, including (without limitation) for quality, stability or performance improvements or to support new hardware. 

   6.2 SDK Integration. If you publicly release a DLSS integration in an end user game or application that presents material stability, performance, image quality, or other technical issues impacting the user experience, you will work to quickly address the integration issues. In the case issues are not addressed, NVIDIA reserves the right, as a last resort, to temporarily disable the DLSS integration until the issues can be fixed.

   7. Marketing.
   7.1 Marketing Activities. Your license to the SDK(s) under the Agreement is subject to your compliance with the following marketing terms:
   (a)  Identification by You in the DLSS SDK, NGX SDK, RTX Video SDK or Dynamic Vibrance SDK.  During the term of the Agreement, NVIDIA agrees that you may identify NVIDIA on your websites, printed collateral, trade-show displays and other retail packaging materials, as the supplier of the DLSS SDK,  NGX SDK, RTX Video SDK or Dynamic Vibrance SDK for the applications that were developed with use of such SDKs, provided that all such references to NVIDIA will be subject to NVIDIA's prior review and written approval, which will not be unreasonably withheld or delayed. 
   (b)  NVIDIA Trademark Placement in Applications with the DLSS SDK, NGX SDK, or RTX Video SDK.  For applications that incorporate the DLSS SDK or NGX SDK or portions thereof, you must attribute the use of the applicable SDK and include the NVIDIA Marks on splash screens, in the about box of the application (if present), and in credits for game applications.
   (c) NVIDIA Trademark Placement in Applications with a licensed SDK, other than the DLSS SDK, RTX Video SDK or NGX SDK. For applications that incorporates and/or makes use of a licensed SDK, other than the DLSS SDK, RTX Video SDK or NGX SDK, you must attribute the use of the applicable SDK and include the NVIDIA Marks on the credit screen for applications that have such credit screen, or where a credit screen is not present prominently in end user documentation for the application. 
   (d) Identification by NVIDIA in the DLSS SDK, NGX SDK, RTX Video SDK or Dynamic Vibrance SDK.  You agree that NVIDIA may identify you on NVIDIA's websites, printed collateral, trade-show displays, and other retail packaging materials as an individual or entity that produces products and services which incorporate the DLSS SDK, NGX SDK, RTX Video SDK or Dynamic Vibrance SDK as applicable. To the extent that you provide NVIDIA with input or usage requests with regard to the use of your logo or materials, NVIDIA will use commercially reasonable efforts to comply with such requests.  For the avoidance of doubt, NVIDIA’s rights pursuant to this section shall survive any expiration or termination of the Agreement with respect to existing applications which incorporate the DLSS SDK, RTX Video SDK or NGX SDK.
   (e) Applications Marketing Material in the DLSS SDK, NGX SDK, RTX Video SDK or Dynamic Vibrance SDK. You may provide NVIDIA with screenshots, imagery, and video footage of applications representative of your use of the NVIDIA DLSS SDK or NGX SDKs in your application (collectively, “Assets”). You hereby grant to NVIDIA the right to create and display self-promotional demo materials using the Assets, and after release of the application to the public to distribute, sub-license, and use the Assets to promote and market the NVIDIA RTX SDKs.  To the extent you provide NVIDIA with input or usage requests with regard to the use of your logo or materials, NVIDIA will use commercially reasonable efforts to comply with such requests.  For the avoidance of doubt, NVIDIA’s rights pursuant to this section shall survive any termination of the Agreement with respect to applications which incorporate the NVIDIA RTX SDK.

   7.2 Trademark Ownership and Licenses. Trademarks are owned and licenses as follows:
   (a)	Ownership of Trademarks.  Each party owns the trademarks, logos, and trade names (collectively "Marks") for their respective products or services, including without limitation in applications, and the NVIDIA RTX SDKs. Each party agrees to use the Marks of the other only as permitted in this exhibit.

   (b)	Trademark License to NVIDIA.  You grant to NVIDIA a non-exclusive, non-sub licensable, non-transferable (except as set forth in the assignment provision of the Agreement), worldwide license to refer to you and your applications, and to use your Marks on NVIDIA's marketing materials and on NVIDIA's website (subject to any reasonable conditions of you) solely for NVIDIA’s marketing activities set forth in this exhibit Sections 7 (d)-(e) above.  NVIDIA will follow your specifications for your Marks as to style, color, and typeface as reasonably provided to NVIDIA.

   (c)	Trademark License to You.  NVIDIA grants to you a non-exclusive, non-sub licensable, non-transferable (except as set forth in the assignment provision of the Agreement), worldwide license, subject to the terms of this exhibit and the Agreement, to use NVIDIA RTX™, NVIDIA GeForce RTX™ in combination with GeForce products, and/or NVIDIA Quadro RTX™  in combination with Quadro products (collectively, the “NVIDIA Marks”) on your marketing materials and on your website (subject to any reasonable conditions of NVIDIA) solely for your marketing activities set forth in this exhibit Sections 7.1 (a)-(c) above.  For the avoidance of doubt, you will not and will not permit others to use any NVIDIA Mark for any other goods or services, or in a way that tarnishes, degrades, disparages or reflects adversely any of the NVIDIA Marks or NVIDIA’s business or reputation, or that dilutes or otherwise harms the value, reputation or distinctiveness of or NVIDIA’s goodwill in any NVIDIA Mark. In addition to the termination rights set forth in the Agreement, NVIDIA may terminate this trademark license at any time upon written notice to you. You will follow NVIDIA's use guidelines and specifications for NVIDIA's Marks as to style, color and typeface as provided in NVIDIA Marks and submit a sample of each proposed use of NVIDIA's Marks at least two (2) weeks prior to the desired implementation of such use to obtain NVIDIA's prior written approval (which approval will not be unreasonably withheld or delayed).  If NVIDIA does not respond within ten (10) business days of your submission of such sample, the sample will be deemed unapproved. All goodwill associated with use of NVIDIA Marks will inure to the sole benefit of NVIDIA. For the RTX Video SDK, contact NVIDIA at nvidia-rtx-video-sdk-license-questions@nvidia.com.

   7.3 Use Guidelines. Use of the NVIDIA Marks is subject to the following guidelines:
   (a)	Business Practices.  You covenant that you will: (a)  conduct business with respect to NVIDIA’s products in a manner that reflects favorably at all times on the good name, goodwill and reputation of such products; (b) avoid deceptive, misleading or unethical practices that are detrimental to NVIDIA, its customers, or end users; (c) make no false or misleading representations with regard to NVIDIA or its products; and (d) not publish or employ or cooperate in the publication or employment of any misleading or deceptive advertising or promotional materials.

   (b)	No Combination Marks or Similar Marks.  You agree not to (a) combine NVIDIA Marks with any other content without NVIDIA’s prior written approval, or (b) use any other trademark, trade name, or other designation of source which creates a likelihood of confusion with NVIDIA Marks.

   (v. March 14, 2024)
   ```
