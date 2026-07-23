/*
* Copyright (C) 2024 Texas Instruments Incorporated
*
* All rights reserved not granted herein.
* Limited License.  
*
* Texas Instruments Incorporated grants a world-wide, royalty-free, 
* non-exclusive license under copyrights and patents it now or hereafter 
* owns or controls to make, have made, use, import, offer to sell and sell ("Utilize")
* this software subject to the terms herein.  With respect to the foregoing patent 
* license, such license is granted  solely to the extent that any such patent is necessary 
* to Utilize the software alone.  The patent license shall not apply to any combinations which 
* include this software, other than combinations with devices manufactured by or for TI ("TI Devices").  
* No hardware patent is licensed hereunder.
*
* Redistributions must preserve existing copyright notices and reproduce this license (including the 
* above copyright notice and the disclaimer and (if applicable) source code license limitations below) 
* in the documentation and/or other materials provided with the distribution
*
* Redistribution and use in binary form, without modification, are permitted provided that the following
* conditions are met:
*
*	* No reverse engineering, decompilation, or disassembly of this software is permitted with respect to any 
*     software provided in binary form.
*	* any redistribution and use are licensed by TI for use only with TI Devices.
*	* Nothing shall obligate TI to provide you with source code for the software licensed and provided to you in object code.
*
* If software source code is provided to you, modification and redistribution of the source code are permitted 
* provided that the following conditions are met:
*
*   * any redistribution and use of the source code, including any resulting derivative works, are licensed by 
*     TI for use only with TI Devices.
*   * any redistribution and use of any object code compiled from the source code and any resulting derivative 
*     works, are licensed by TI for use only with TI Devices.
*
* Neither the name of Texas Instruments Incorporated nor the names of its suppliers may be used to endorse or 
* promote products derived from this software without specific prior written permission.
*
* DISCLAIMER.
*
* THIS SOFTWARE IS PROVIDED BY TI AND TI'S LICENSORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, 
* BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. 
* IN NO EVENT SHALL TI AND TI'S LICENSORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
* CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, 
* OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, 
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
* POSSIBILITY OF SUCH DAMAGE.
*/

/* Standard Include Files. */
#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>

/* mmWave SDK drivers/common Include Files */
#include <ti/board/antenna_geometry.h>

#ifdef SOC_XWR16XX
/**
 * @brief Antenna geometry for XWR1642
 *
 */
ANTDEF_AntGeometry gAntDef_default = {
    .txAnt = {
        { 0, 0 },
        { 4, 0 } },
    .rxAnt = { { 0, 0 }, { 1, 0 }, { 2, 0 }, { 3, 0 } }
};
#else
/**
 * @brief Antenna geometry for IWR6843 AOP
 *
 */
ANTDEF_AntGeometry gAntDef_IWR6843AOP = {
    .txAnt = {
        { 0, 0 },
        { 2, 2 },
        { 0, 2 } },
    .rxAnt = { { 1, 1 }, { 1, 0 }, { 0, 1 }, { 0, 0 } }
};
ANTDEF_AntGeometry gAntDef_IWR6843ODS = {
    .txAnt = {
        { 0, 0 },
        { 2, 0 },
        { 2, 2 } },
    .rxAnt = { { 0, 0 }, { 0, 1 }, { 1, 1 }, { 1, 0 } }
};
/**
 * @brief Antenna geometry for standard EVM boards: XWR1843, XWR6843
 *
 */
ANTDEF_AntGeometry gAntDef_default = {
    .txAnt = {
        { 0, 1 },
        { 2, 0 },
        { 4, 1 } },
    .rxAnt = { { 0, 0 }, { 1, 0 }, { 2, 0 }, { 3, 0 } }
};
/**
 * @brief Antenna geometry for AWR1843 AOP
 *
 */
ANTDEF_AntGeometry gAntDef_AWR1843AOP = {
    .txAnt = {
        { 0, 0 },
        { 0, 1 },
        { 0, 2 } },
    .rxAnt = { { 3, 0 }, { 2, 0 }, { 1, 0 }, { 0, 0 } }
};
#endif

#if defined(SOC_XWR68XX)
#ifdef ISK
float gAntennaSpacing = 2.5e-3;
#elif defined(ODS)
float gAntennaSpacing = 2.426e-3;
#elif defined(AOP)
float gAntennaSpacing = 2.5e-3;
#elif defined(DEFAULT_ANT_DESIGN)
float gAntennaSpacing = 2.5e-3;
#else
float gAntennaSpacing = -1; /* Antenna spacing value based correction factor set to default */
#endif
#else
float gAntennaSpacing = -1; /* Antenna spacing value based correction factor set to default or not applicable for other SOCs */
#endif
