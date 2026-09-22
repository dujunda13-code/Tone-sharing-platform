/** 预览图中的连续半透明声波，仅作装饰。 */
export function AtmosphericWave({ className = "" }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 1000 260" preserveAspectRatio="none" aria-hidden="true" focusable="false">
      <defs>
        <linearGradient id="wave-mint-sky" x1="0" x2="1">
          <stop stopColor="#a7e5d3" stopOpacity=".1" />
          <stop offset=".42" stopColor="#a7e5d3" stopOpacity=".6" />
          <stop offset="1" stopColor="#a8c8e8" stopOpacity=".13" />
        </linearGradient>
        <linearGradient id="wave-rose-lavender" x1="0" x2="1">
          <stop stopColor="#e8b8c4" stopOpacity=".1" />
          <stop offset=".48" stopColor="#c8b8e0" stopOpacity=".55" />
          <stop offset="1" stopColor="#c8b8e0" stopOpacity=".1" />
        </linearGradient>
        <linearGradient id="wave-sky-mint" x1="0" x2="1">
          <stop stopColor="#a8c8e8" stopOpacity=".04" />
          <stop offset=".54" stopColor="#a8c8e8" stopOpacity=".38" />
          <stop offset="1" stopColor="#a7e5d3" stopOpacity=".12" />
        </linearGradient>
      </defs>
      <path fill="url(#wave-mint-sky)" d="M0 138 C55 154 80 92 125 110 S174 176 216 146 S263 55 308 83 S348 165 394 147 S463 86 511 110 S570 178 620 144 S692 93 748 113 S822 168 877 129 S950 100 1000 115 L1000 238 C920 182 875 195 817 213 S713 240 650 209 S565 169 507 196 S416 239 354 203 S275 165 214 194 S105 230 0 184 Z" />
      <path fill="url(#wave-rose-lavender)" d="M0 126 C80 87 103 163 158 149 S230 76 291 99 S352 182 409 156 S490 42 550 78 S604 182 667 156 S737 75 797 100 S878 173 933 130 S969 97 1000 101 L1000 178 C944 227 890 207 833 181 S742 146 689 198 S594 234 544 178 S467 124 415 205 S317 208 272 169 S198 132 145 196 S55 196 0 174 Z" />
      <path fill="url(#wave-sky-mint)" d="M0 159 C69 118 94 183 149 166 S242 106 296 128 S363 193 421 169 S497 105 554 125 S620 190 675 168 S745 119 806 138 S904 198 1000 148 L1000 222 C908 177 860 242 802 211 S716 174 659 226 S573 194 528 169 S449 229 393 213 S318 156 258 196 S173 229 117 198 S43 179 0 204 Z" />
    </svg>
  );
}
