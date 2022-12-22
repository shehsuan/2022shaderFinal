// Author:CMH
// Title: Basic Raymarching_2(normal, camera) 
// Reference: 20220414_glsl Breathing Circle_v5A(BRDF).qtz

#ifdef GL_ES
precision mediump float;
#endif

uniform vec2 u_resolution;
uniform vec2 u_mouse;
uniform float u_time;

// SDF framework by Inigo Quilez:
// https://www.shadertoy.com/view/Xds3zN
//

#define PI 3.141592654
#define TWOPI 6.283185308
vec2 SphereMap( vec3 ray){		//ray mapping to UV
   vec2 st;
   ray=normalize(ray);
   float radius=length(ray);
   st.y = acos(ray.y/radius) / PI;
   if (ray.z >= 0.0) st.x = acos(ray.x/(radius * sin(PI*(st.y)))) / TWOPI;
   else st.x = 1.0 - acos(ray.x/(radius * sin(PI*(st.y)))) / TWOPI;
   return st;
}


vec4 warpcolor(in vec2 uv, float t){   //Normalized uv[0~1]
    		float strength = 0.4;
		vec3 col = vec3(0);
		//pos coordinates (from -1 to 1)
		vec2 pos = uv*2.0-1.0;
            
		//請小心！QC迴圈最好使用int index，float index有可能錯誤！
		for(int i = 1; i < 7; i++){ 
		pos.x += strength * sin(2.0*t+float(i)*1.5 * pos.y)+t*0.5;
		pos.y += strength * cos(2.0*t+float(i)*1.5 * pos.x);}

		//Time varying pixel colour
		col += 0.5 + 0.5*cos(t+pos.xyx+vec3(0,2,4));
		//Gamma
		col = pow(col, vec3(0.4545));
		return vec4(col,1.0) ;
}

vec3 warpSky(vec3 e){
     vec2 ST=SphereMap(e);
     vec4 color = warpcolor(ST, u_time*0.1);
    return color.xyz;
}


vec3 phong(vec3 p, vec3 n, vec3 v){
    vec3 final=vec3(0.0);
    vec3 diffuse,specular,ambient;
    ambient=vec3(0.305,0.049,0.049);
        
    {//light1    
    vec3 light_pos= vec3(5.000,1.000,2.000);
    vec3 light_col= vec3(0.955,0.819,0.231);
    vec3 l=normalize(light_pos-p); //光線向量
    vec3 r=normalize(reflect(-l,n));
    float ka=0.1, ks=1.0, kd=1.0;
    float shininess=8.0;        
    diffuse=vec3(kd*dot(l, n));
    specular=vec3(ks* pow(max(dot(r, v),0.0), shininess)) ;    
    final+=(diffuse+specular)*light_col;
    }
    
    {//light2
    vec3 light_pos= vec3(-5.000,1.000,2.000);
    vec3 light_col= vec3(0.137,0.955,0.646);
    vec3 l=normalize(light_pos-p); //光線向量
    vec3 r=normalize(reflect(-l,n));
    float ka=0.1, ks=0.5, kd=1.0;
    float shininess=6.0;    
    diffuse=vec3(kd*dot(l, n));
    specular=vec3(ks* pow(max(dot(r, v),0.0), shininess)) ;    
    final+=(diffuse+specular)*light_col;//動手腳
    }
    
    final = final+ambient;
    
    vec3 refl=reflect(-v,n);
    vec3 refl_color=warpSky(refl);    
    float F=1.0-1.0*dot(n,v); //edge=0 center=1
    final = mix(final,refl_color, F);
    return final;
    
}

vec2 boxIntersect(in vec3 ro, in vec3 rd, in vec3 rad) {
    vec3 m = 1./rd;
    vec3 n = m*ro;
    vec3 k = abs(m)*rad;
    
    vec3 t1 = -n - k;
    vec3 t2 = -n + k;
    
    float tN = max(max(t1.x, t1.y), t1.z);
    float tF = min(min(t2.x, t2.y), t2.z);
    
    if(tN > tF || tF < .0) return vec2(-1);
    
    return vec2(tN, tF);
}

float smin(float a, float b, float k) {
    float h = clamp(.5 + .5*(b - a)/k, .0, 1.);
    return mix(b, a, h) - k * h * (1. - h);
}

float udRoundBox(vec3 p, vec3 b, float r) {
    return length(max(abs(p)-b, .0)) -r;
}

float sdCapsuleF(vec3 p, vec3 a, vec3 b, const float r0, const float r1, const float f) {
    vec3 d = b -a;
    float h = length(d);
    d = normalize(d);
    float t=dot(p-a, d);
    float th = t/h;
    return distance(a+clamp(t,0.,h)*d, p)-mix(r0, r1, th) * 
           max(0., 1.+f-f*4.*abs(th-.5)*abs(th -.5));
}

float sdCapsule(vec3 p, vec3 a, vec3 b, const float r0, const float r1) {
    vec3 d = b -a;
    float h = length(d);
    d = normalize(d);
    float t=clamp(dot(p-a, d), 0., h);
    return distance(a+t*d, p) -mix(r0, r1, t/h);
}

float mapHand(in vec3 p) {
    float sph = length(p) - .1;
    if (sph > .1) return sph; //  bounding sphere
    
    const float s = 1.175;
    float d = udRoundBox(p, vec3(.0175/s + p.y * (.25/s), .035/s + p.x * (.2/s), 0.), .01);
    d = smin(d, min(sdCapsule(p, vec3(.025, .0475, 0)/s, vec3(.028, .08, .02)/s, .01/s, .0075/s), 
                    sdCapsule(p, vec3(.028, .08, .02)/s, vec3(.03, 0.1, .06)/s, .0075/s, .007/s)), .0057);
    d = smin(d, min(sdCapsule(p, vec3(.01, .0425, 0)/s, vec3(.008, .07, .025)/s, .009/s, .0075/s), 
                    sdCapsule(p, vec3(.008, .07, .025)/s, vec3(.008, .085, .065)/s, .0075/s, .007/s)), .0057);
    d = smin(d, min(sdCapsule(p, vec3(-.01, .04, 0)/s, vec3(-.012, .065, .028)/s, .009/s, .0075/s), 
                    sdCapsule(p, vec3(-.012, .065, .028)/s, vec3(-.012, .07, .055)/s, .0075/s, .007/s)), .0057);
    d = smin(d, min(sdCapsule(p, vec3(-.025, .035, 0)/s, vec3(-.027, .058, .03)/s, .009/s, .0075/s), 
                    sdCapsule(p, vec3(-.027, .058, .03)/s, vec3(-.028, .06, .05)/s, .0075/s, .007/s)), .0057);
    return d;
}

float mapWoman(in vec3 pos) {
    const float f0 = .075;
    const float f1 = .2;
    const float f2 = .275;
    
    vec3 ph = pos;
    
    if (pos.x < 0.) {
        ph += vec3(.11, -.135, .2);
        ph = mat3(-0.8674127459526062, -0.49060970544815063, 0.08304927498102188, 0.22917310893535614, -0.5420454144477844, -0.8084964156150818, 0.4416726529598236, -0.6822674870491028, 0.5826116800308228) * ph;
    } else {
        ph.x = -ph.x;
        ph += vec3(.075, -.09, .125);
        ph = mat3(-0.6703562140464783, -0.7417424321174622, 0.020991835743188858, 0.36215442419052124, -0.3517296612262726, -0.8632093667984009, 0.6476624608039856, -0.5710554718971252, 0.5044094920158386) * ph;
    }
    
    float dh = mapHand(ph);
    
    //  right arm
    float d = sdCapsuleF(pos, vec3(0.13, 0.535, -.036), vec3(.09, 0.292, -0.1), .035, .025, f1);
    d = smin(d, sdCapsuleF(pos, vec3(.08, 0.29, -0.1), vec3(-.09, 0.15, -0.17), .03, .02, f0), .0051);
    if (pos.x < 0.) d = smin(d, dh, .015);
    
    //  left arm
    float d1 = sdCapsuleF(pos, vec3(-0.12, 0.56, .02), vec3(-0.11, 0.325, -.045), .035, .025, f1);
    d1 = smin(d1, sdCapsuleF(pos, vec3(-0.11, 0.315, -.05), vec3(.07, .08, -0.11), .024, .022, f2), .005);
    if (pos.x > 0.) d1 = smin(d1, dh, .015);
    d = min(d1, d);
    
    //  body
    vec3 bp1 = pos;
    bp1 += vec3(0, -.44, -.027);
    bp1 = mat3(0.9761762022972107, 0.033977385610342026, 0.2143024057149887, -0.07553963363170624, 0.9790945649147034, 0.18885889649391174, -0.20340539515018463, -0.20054790377616882, 0.9583353996276855) * bp1;
    float db = udRoundBox(bp1, vec3(.07 + bp1.y*.3, 0.135 -abs(bp1.x)*0.2, 0.), .04);
    
    vec3 bp2 = pos;
    bp2 += vec3(-.032, -.235, -.06);
    bp2 = mat3(0.8958174586296082, -0.37155669927597046, 0.24383758008480072, 0.3379548490047455, 0.9258314967155457, 0.16918234527111053, -0.28861331939697266, -0.0691504031419754, 0.9549453258514404) * bp2;
    db = smin(db, udRoundBox(bp2, vec3(.065 - bp2.y*.25, 0.1, .02 -bp2.y*.13), .04), .03);
    
    db = smin(db, sdCapsule(pos, vec3(0.11, 0.5, -.032), vec3(.05, 0.52, -.015), .04, .035), .01);
    db = smin(db, sdCapsule(pos, vec3(.01, 0.4, -.01), vec3(.01, 0.7, .0), .045, .04), .02);
    
    vec3 bp3 = pos;
    bp3 += vec3(-.005, -.48, .018);
    bp3 = mat3(0.9800665974617004, 0.05107402056455612, 0.19199204444885254, 0, 0.9663899540901184, -0.2570805549621582, -0.19866932928562164, 0.2519560754299164, 0.9471265077590942) * bp3;
    db = smin(db, udRoundBox(bp3, vec3(.056 + bp3.y*.23 , .06, 0.), .04), .01);
    
    d = smin(d, db, .01);
    
    //  right leg
    float d2 = sdCapsuleF(pos, vec3(0.152, 0.15, .05), vec3(-.03, 0.43, -.08), .071, .055, f2);
    d2 = smin(d2, sdCapsuleF(pos, vec3(0.14, .08, .05), vec3(-.01, 0.23, -.02), .05, .02, f1), .075);
    d = min(d, d2);
    float d3 = sdCapsuleF(pos, vec3(-.03, 0.43, -.084), vec3(.055, .04, -.04), .053, .02, f0);
    d3 = smin(d3, sdCapsuleF(pos, vec3(-.0, 0.35, -.05), vec3(.025, 0.2, -.03), .04, .02, f2), .05);
    d = min(d, d3);
    
    //  left leg
    d = min(d, sdCapsuleF(pos, vec3(-.02, 0.12, 0.1), vec3(-0.145, .08, -0.17), .07, .055, f2));
    float d4 = sdCapsuleF(pos, vec3(-0.145, .08, -0.17), vec3(0.205, .02, -0.09), .05, .0185, f0);
    d4 = smin(d4, sdCapsuleF(pos, vec3(-.05, .085, -0.145), vec3(.05, .03, -.09), .035, .03, f2), .0075);
    
    //  right feet
    float d6 = distance(pos, vec3(.0, .0, -0.1)) -.1; //  bounding sphere
    if(d6 < 0.1) {
        d = min(d, sdCapsule(pos, vec3(.03, .03, -.08), vec3(.031, .01, -0.146), .015, .005));
        d = min(d, sdCapsule(pos, vec3(.02, .03, -.08), vec3(.018, .01, -0.1505), .015, .006));
        d = min(d, sdCapsule(pos, vec3(.00, .03, -.08), vec3(.005, .01, -0.1525), .015, .007));
        d = min(d, sdCapsule(pos, vec3(-.01, .03, -.08), vec3(-.014, .01, -0.1575), .015, .01));
    } else {
        d = min(d6, d);
    }
    
    //  left feet
    float d5 = distance(pos, vec3(0.25, .025, -0.1)) -.12; //  bounding sphere
    if(d5 < 0.1) {
        d5 = sdCapsule(pos, vec3(0.2, .035, -.075), vec3(0.3, .01, -.09), .035, .02);
        d5 = smin(d5, sdCapsule(pos, vec3(0.31, .035, -.0975), vec3(0.1, .01, -0.10), .015, .02), .02);
        d5 = smin(d5, sdCapsule(pos, vec3(0.31, .035, -.0975), vec3(0.355, .034, -0.10), .015, .01), .005);
        d5 = min(d5, sdCapsule(pos, vec3(0.31, .022, -.0875), vec3(0.335, .022, -.09), .02, .01));
    }
    d4 = smin(d4, d5, .025);
    d = min(d, d4);
    
    //  hair
    vec3 hp = pos;
    hp.x += smoothstep(.55, .45, pos.y)*.035;
    hp.z *= 1.9 - .8 * pos.y;
    hp.yz -= 2.*pos.x*pos.x;
    float h = sdCapsule(hp, vec3(.0, 0.725, -.02), vec3(-.02, 0.415, .0), .094, .085);
    //h = smin(h, sdCapsule(hp, vec3(.0, 0.725, -.02), vec3(.06, 0.705, -.05), .085, .095), .02);
    h = smin(h, sdCapsule(hp, vec3(.0, 0.725, -.02), vec3(.06, 0.705, -.05), .05, .06), .02);
    h = max(-(pos.y - abs(fract(pos.x*90.) -.5)*0.1 -.14 - smoothstep(-0.2, 0.1, pos.x)*.5), h);
    
    return (h < d) ? h : d;
}
vec3 mapMushroom( vec3 p );   
//=== distance functions ===
float sdSphere( vec3 p, float s )
{
    return length(p)-s;
}
float sdBox( vec3 p, vec3 b )
{
  vec3 d = abs(p) - b;
  return min(max(d.x,max(d.y,d.z)),0.0) + length(max(d,0.0));
}
float sdTorus( vec3 p, vec2 t )
{
  vec2 q = vec2(length(p.xy)-t.x,p.z);
  return length(q)-t.y;
}

float map(in vec3 p)
{
///return sdSphere(p+vec3(0.,0.,0.0), 0.5);
//return sdTorus(p+vec3(0.,0.,0.0),vec2(0.4,0.2));
//return sdBox(p+vec3(0.0,0.0,0.0), vec3(0.4, 0.4, 0.4));
    //return mapWoman(p+vec3(0.,0.4,.0));
    return mapMushroom(p*1.5+vec3(0.,0.0,0.5)).x;
}

//=== gradient functions ===
vec3 gradient( in vec3 p ) //尚未normalize
{
	const float d = 0.001;
	vec3 grad = vec3(map(p+vec3(d,0,0))-map(p-vec3(d,0,0)),
                     map(p+vec3(0,d,0))-map(p-vec3(0,d,0)),
                     map(p+vec3(0,0,d))-map(p-vec3(0,0,d)));
	return grad;
}


// === raytrace functions===
float trace(vec3 o, vec3 r, out vec3 p)
{
float d=0.0, t=0.0;
for (int i=0; i<32; ++i)
{
	p= o+r*t;
	d=map(p);
	if(d<0.0) break;
	t += d*0.6; //影響輪廓精準程度   //(思函)香菇輪廓
	}
return t;
}


//=== sky ===
float fbm(in vec2 uv);
vec3 getSkyFBM(vec3 e) {	//二維雲霧
	vec3 f=e;
	float m = 2.0 * sqrt(f.x*f.x + f.y*f.y + f.z*f.z);
	vec2 st= vec2(-f.x/m + .5, -f.y/m + .5);
	//vec3 ret=texture2D(iChannel0, st).xyz;
	float fog= fbm(0.6*st+vec2(-0.2*u_time, -0.02*u_time))*0.5+0.3;
    return vec3(fog);
}

vec3 sky_color(vec3 e) {	//漸層藍天空色
    /* === comment by 蘇蘇：===
                2.0 * e.y 是讓背景顏色對比明顯一點，淺色的更白，深色的更黑
                0.4 是定義最淺的顏色
    */
    e.y = max(2.0 * e.y,0.4);
    
    vec3 ret;
    ret.x = pow(1.0-e.y,3.0);
    ret.y = pow(1.0-e.y, 1.2);
    ret.z = 0.8+(1.0-e.y)*0.3;    
    return ret;
}

vec3 getSkyALL(vec3 e)
{	
	return getSkyFBM(e);
}

//=== camera functions ===
mat3 setCamera( in vec3 ro, in vec3 ta, float cr )
{
	vec3 cw = normalize(ta-ro);
	vec3 cp = vec3(sin(cr), cos(cr),0.0);
	vec3 cu = normalize( cross(cw,cp) );
	vec3 cv = normalize( cross(cu,cw) );
    return mat3( cu, cv, cw );
}

// math
mat3 fromEuler(vec3 ang) {
    vec2 a1 = vec2(sin(ang.x),cos(ang.x));
    vec2 a2 = vec2(sin(ang.y),cos(ang.y));
    vec2 a3 = vec2(sin(ang.z),cos(ang.z));
    vec3 m0 = vec3(a1.y*a3.y+a1.x*a2.x*a3.x,a1.y*a2.x*a3.x+a3.y*a1.x,-a2.y*a3.x);
    vec3 m1 = vec3(-a2.y*a1.x,a1.y*a2.y,a2.x);
    vec3 m2 = vec3(a3.y*a1.x*a2.x+a1.y*a3.x,a1.x*a3.x-a1.y*a3.y*a2.x,a2.y*a3.y);
    return mat3(m0, m1, m2);
}


/* === comment by 蘇蘇：===
            宣告 function
*/
float calcAO( in vec3 pos, in vec3 nor );
float noise_3(in vec3 p);
vec3 FlameColour(float f);
vec3 normalMap(vec3 p, vec3 n);
/* === end === */



// ================
void main()
{
vec2 uv = gl_FragCoord.xy/u_resolution.xy;
uv = uv*2.0-1.0;
uv.x*= u_resolution.x/u_resolution.y;
uv.y*=1.0;//校正 預設值uv v軸朝下，轉成v軸朝上相同於y軸朝上為正
vec2 mouse=(u_mouse.xy/u_resolution.xy)*2.0-1.0;

// camera option1  (模型應在原點，適用於物件)
		vec3 CameraRot=vec3(0.2, mouse.y+.2, mouse.x+.2);  //(思函)調整鏡頭
        //vec3 CameraRot=vec3(0.0, mouse.y, mouse.x);  //(原始)滑鼠鏡頭
        //vec3 CameraRot=vec3(0.0, 0.2 ,0.0);  //(思函)調整鏡頭
	vec3 ro= vec3(1.0, 0.0,1.0)*fromEuler(CameraRot);//CameraPos;
	vec3 ta =vec3(0.5, 0.5, 0.0); //TargetPos; //vec3 ta =float3(CameraDir.x, CameraDir.z, CameraDir.y);//UE座標Z軸在上
	mat3 ca = setCamera( ro, ta, 0.0 );
	vec3 RayDir = ca*normalize(vec3(uv, 1.0));//z值越大，zoom in! 可替換成iMouse.z
	vec3 RayOri = ro;

// camera option2 (攝影機在原點，適用於場景)
/*	
	vec3 CameraRot=vec3(0.0, -iMouse.y, -iMouse.x);
	vec3 RayOri= vec3(0.0, 0.0, 0.0);	//CameraPos;
	vec3 RayDir = normalize(vec3(uv, -1.))*fromEuler(CameraRot);
*/
	

    
    vec3 p,n;
	float t = trace(RayOri, RayDir, p); //position
	n=normalize(gradient(p)); //normal
    


    /* === comment by 蘇蘇：===
                黑色凹凸的程式碼
                p*10.0 控制斑點數量
                bump*1.0 控制斑點大小
    */
//RAYMARCHING_3
    vec3 bump=normalMap( p*5.0, n);
    n=n+bump*2.; //（可以使用+_*去調整bump來改變特效 bump可以讓質地更有凹凸面）
	/* === end === */



    /* === comment by 蘇蘇：===
                乳牛斑點的程式碼
                p*10.0 控制斑點數量
                noise_3(...)*2.0-1.0 控制斑點大小
                10. * (noise_3(...)...) 控制斑點的顏色濃度
    */
//COW SPOT
    /*
    float displacement = 10. * (noise_3(p*10.0)*2.0-1.0);
 	p += n*displacement;
 	n=normalize(gradient(p));
    //n=n+bump*0.5; //（可以使用+_*去調整bump來改變特效 bump可以讓質地更有凹凸面）
    */
	/* === end === */


    
    /* === comment by 蘇蘇：===
                不管老師的黑色凹凸
                或是私函的乳牛斑點
                都要有這段程式碼
    */
    float edge= dot(RayDir, n);  //_+黑跟白會轉換
    //edge = step(0.2,edge);  //黑邊
    edge = smoothstep(-0.2,0.4,edge);
    
    vec3 result=n;
    vec3 ao = vec3(calcAO(p,n));
    //result = FlameColour(calcAO(p,n)); //FlameColour
    result = vec3(edge); //邊角是白色，內裡是黑色
    //result = phong(p,n,-RayDir) * ao;
    /* === end === */
    
    
    
    
//SHADING
    /* === comment by 蘇蘇：===
    			最後一個是我把煙霧特效複製過來的
                其他都是原本就有
                可以自己試試看加上不同的特效
                (把 // 拿掉，= 試著換成 +=, *=, 或 =)
    */
    
    //result=normalize(p);
    //result=(n);
    //result=vec3(t*0.612);
    //result=vec3(1.0-exp(-t*0.612));
    //result=getSkyALL(reflect(RayDir,n)); //p or n
    //result += fbm(0.6*uv+vec2(-0.2*u_time, -0.02*u_time))*0.5+0.3;
	/* === end === */
    

    
//HDR環境貼圖
    /* === comment by 蘇蘇：===
                讓背景顏色反過來，從天空變成深海
    */
	vec3 BG=getSkyALL(1.0 - RayDir);	   //或getSkyFBM(RayDir)
    BG = getSkyFBM(n);
    /* === end === */

    

//亂數作用雲霧(二維)
float fog= fbm(0.6*uv+vec2(-0.2*u_time, -0.02*u_time))*0.5+0.3;
//vec3 fogFBM=getSkyFBM(reflect(RayDir,n));

//gl_FragColor = vec4(vec3(result),1.0);    

//（思函）背景雜訊
//if(t<3.5) gl_FragColor = vec4(vec3(result),1.0); else gl_FragColor = vec4(BG,1.0);  

//（思函）背景雜訊
if(t>=0. && t<=3.5) gl_FragColor = vec4(vec3(result),1.0); else gl_FragColor = vec4(BG,1.0);
}




















//=== 3d noise functions ===
float hash11(float p) {
    return fract(sin(p * 727.1)*43758.5453123);
}
float hash12(vec2 p) {
	float h = dot(p,vec2(127.1,311.7));	
    return fract(sin(h)*43758.5453123);
}
vec3 hash31(float p) {
	vec3 h = vec3(1275.231,4461.7,7182.423) * p;	
    return fract(sin(h)*43758.543123);
}
// 3d noise
float noise_3(in vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);	
	vec3 u = f*f*(3.0-2.0*f);    
    vec2 ii = i.xy + i.z * vec2(5.0);
    float a = hash12( ii + vec2(0.0,0.0) );
	float b = hash12( ii + vec2(1.0,0.0) );    
    float c = hash12( ii + vec2(0.0,1.0) );
	float d = hash12( ii + vec2(1.0,1.0) ); 
    float v1 = mix(mix(a,b,u.x), mix(c,d,u.x), u.y);    
    ii += vec2(5.0);
    a = hash12( ii + vec2(0.0,0.0) );
	b = hash12( ii + vec2(1.0,0.0) );    
    c = hash12( ii + vec2(0.0,1.0) );
	d = hash12( ii + vec2(1.0,1.0) );
    float v2 = mix(mix(a,b,u.x), mix(c,d,u.x), u.y);        
    return max(mix(v1,v2,u.z),0.0);
}
//=== glow functions ===
float glow(float d, float str, float thickness){
    return thickness / pow(d, str);
}

//=== 2d noise functions ===
vec2 hash2( vec2 x )			//亂數範圍 [-1,1]
{
    const vec2 k = vec2( 0.3183099, 0.3678794 );
    x = x*k + k.yx;
    return -1.0 + 2.0*fract( 16.0 * k*fract( x.x*x.y*(x.x+x.y)) );
}
float gnoise( in vec2 p )		//亂數範圍 [-1,1]
{
    vec2 i = floor( p );
    vec2 f = fract( p );
	
    vec2 u = f*f*(3.0-2.0*f);

    return mix( mix( dot( hash2( i + vec2(0.0,0.0) ), f - vec2(0.0,0.0) ), 
                     	    dot( hash2( i + vec2(1.0,0.0) ), f - vec2(1.0,0.0) ), u.x),
                	     mix( dot( hash2( i + vec2(0.0,1.0) ), f - vec2(0.0,1.0) ), 
                     	    dot( hash2( i + vec2(1.0,1.0) ), f - vec2(1.0,1.0) ), u.x), u.y);
}
float fbm(in vec2 uv)		//亂數範圍 [-1,1]
{
	float f;				//fbm - fractal noise (4 octaves)
	mat2 m = mat2( 1.6,  1.2, -1.2,  1.6 );
	f   = 0.5000*gnoise( uv ); uv = m*uv;		  
	f += 0.2500*gnoise( uv ); uv = m*uv;
	f += 0.1250*gnoise( uv ); uv = m*uv;
	f += 0.0625*gnoise( uv ); uv = m*uv;
	return f;
}




// Created by inigo quilez - iq/2017
// I share this piece (art and code) here in Shadertoy and through its Public API, only for educational purposes. 
// You cannot use, sell, share or host this piece or modifications of it as part of your own commercial or non-commercial product, website or project.
// You can share a link to it or an unmodified screenshot of it provided you attribute "by Inigo Quilez, @iquilezles and iquilezles.org". 
// If you are a teacher, lecturer, educator or similar and these conditions are too restrictive for your needs, please contact me and we'll work it out.


#define MAT_MUSH_HEAD 1.0
#define MAT_MUSH_NECK 2.0
#define MAT_LADY_BODY 3.0
#define MAT_LADY_HEAD 4.0
#define MAT_LADY_LEGS 5.0
#define MAT_GRASS     6.0
#define MAT_GROUND    7.0
#define MAT_MOSS      8.0
#define MAT_CITA      9.0

//vec2  hash2( vec2 p ) { p=vec2(dot(p,vec2(127.1,311.7)),dot(p,vec2(269.5,183.3))); return fract(sin(p)*18.5453); }
vec3  hash3( float n ) { return fract(sin(vec3(n,n+1.0,n+2.0))*vec3(338.5453123,278.1459123,191.1234)); }
float dot2(in vec2 p ) { return dot(p,p); }
float dot2(in vec3 p ) { return dot(p,p); }

vec2 sdLine( in vec2 p, in vec2 a, in vec2 b )
{
    vec2 pa = p - a;
    vec2 ba = b - a;
    float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
    return vec2( length(pa-h*ba), h );
}
vec2 sdLine( in vec3 p, in vec3 a, in vec3 b )
{
    vec3 pa = p - a;
    vec3 ba = b - a;
    float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
    return vec2( length(pa-h*ba), h );
}
vec2 sdLineOri( in vec3 p, in vec3 b )
{
    float h = clamp( dot(p,b)/dot(b,b), 0.0, 1.0 );
    
    return vec2( length(p-h*b), h );
}
vec2 sdLineOriY( in vec3 p, in float b )
{
    float h = clamp( p.y/b, 0.0, 1.0 );
    p.y -= b*h;
    return vec2( length(p), h );
}
float sdEllipsoid( in vec3 pos, in vec3 cen, in vec3 rad )
{
    vec3 p = pos - cen;
    float k0 = length(p/rad);
    float k1 = length(p/(rad*rad));
    return k0*(k0-1.0)/k1;
}
/*
float smin( float a, float b, float k )
{
    float h = max(k-abs(a-b),0.0);
    return min(a, b) - h*h*0.25/k;
}
*/
float smax( float a, float b, float k )
{
    float h = max(k-abs(a-b),0.0);
    return max(a, b) + h*h*0.25/k;
}
vec3 rotateX( in vec3 p, float t )
{
    float co = cos(t);
    float si = sin(t);
    p.yz = mat2(co,-si,si,co)*p.yz;
    return p;
}
vec3 rotateY( in vec3 p, float t )
{
    float co = cos(t);
    float si = sin(t);
    p.xz = mat2(co,-si,si,co)*p.xz;
    return p;
}
vec3 rotateZ( in vec3 p, float t )
{
    float co = cos(t);
    float si = sin(t);
    p.xy = mat2(co,-si,si,co)*p.xy;
    return p;
}

//==================================================


//==================================================



vec3 worldToMushrom( in vec3 pos )
{
    vec3 qos = pos;
    qos.xy = (mat2(60,11,-11,60)/61.0) * qos.xy;
    qos.y += 0.03*sin(3.0*qos.z - 2.0*sin(3.0*qos.x));
    qos.y -= 0.4;
    return qos;
}

vec3 mapMushroom( in vec3 pos )
{
    vec3 res;

    vec3 qos = worldToMushrom(pos);

    {
        // head
        float d1 = sdEllipsoid( qos, vec3(0.0, 1.4,0.0), vec3(0.8,1.0,0.8) );

        // displacement
        float f;
        vec3 tos = qos*0.5;
        f  = 1.00*(sin( 63.0*tos.x+sin( 23.0*tos.z)));
        f += 0.50*(sin(113.0*tos.z+sin( 41.0*tos.x)));
        f += 0.25*(sin(233.0*tos.x+sin(111.0*tos.z)));
        f = 0.5*(f + f*f*f);
        d1 -= 0.0005*f - 0.01;

        // cut the lower half
        float d2 = sdEllipsoid( qos, vec3(0.0, 0.5,0.0), vec3(1.3,1.2,1.3) );
        float d = smax( d1, -d2, 0.1 );
        res = vec3( d, MAT_MUSH_HEAD, 0.0 );
    }


    {
        // stem
        pos.x += 0.3*sin(pos.y) - 0.65;
        float pa = sin( 20.0*atan(pos.z,pos.x) );
        vec2 se = sdLine( pos, vec3(0.0,2.0,0.0), vec3(0.0,0.0,0.0) );
        float tt = 0.25 - 0.1*4.0*se.y*(1.0-se.y);
        float d3 = se.x - tt;
        
        // skirt
        vec2 ros = vec2(length(pos.xz),pos.y);
        se = sdLine( ros, vec2(0.0,1.9), vec2(0.31,1.5) );
        float d4 = se.x - 0.02;//*(1.0-se.y);
        d3 = smin( d3, d4, 0.05);

        d3 += 0.003*pa;
        d3 *= 0.7;

        if( d3<res.x )
            res = vec3( d3, MAT_MUSH_NECK, 0.0 );
    }

    return res;
}




/* === comment by 蘇蘇：===
			乳牛斑點特效
            直接複製貼上過來沒有改
*/
float calcAO( in vec3 pos, in vec3 nor )
{
	float ao = 0.0;

	vec3 v = normalize(vec3(0.7,-0.1,-0.1)); //隨機取樣
	for( int i=0; i<12; i++ )
	{
		float h = abs(sin(float(i)));
		vec3 kv = v + 2.0*nor*max(0.0,-dot(nor,v));
		ao += clamp( map(pos+nor*0.01+kv*h*0.08)*3.0, 0.0, 1.0 );
		v = v.yzx; //if( (i&2)==2) v.yz *= -1.0;
	}
	ao /= 12.0;
	ao = ao + 2.0*ao*ao;
	return clamp( ao*5.0, 0.0, 1.0 );
}
 vec3 hsv2rgb_smooth( in vec3 c )
{
    vec3 rgb = clamp( abs(mod(c.x*6.0+vec3(0.0,4.0,2.0),6.0)-3.0)-1.0, 0.0, 1.0 );

	rgb = rgb*rgb*(3.0-2.0*rgb); // cubic smoothing	

	return c.z * mix( vec3(1.0), rgb, c.y);
}

vec3 hsv2rgb_trigonometric( in vec3 c )
{
    vec3 rgb = 0.5 + 0.5*cos((c.x*6.0+vec3(0.0,4.0,2.0))*3.14159/3.0);

	return c.z * mix( vec3(1.0), rgb, c.y);
}

vec3 FlameColour(float f)
{
	return hsv2rgb_smooth(vec3((f-(2.25/6.))*(1.25/6.),f*1.25+.2,f*.95));
}
/* === end === */



/* === comment by 蘇蘇：===
			黑色凹凸特效
            直接複製貼上過來沒有改
*/
vec3 smoothSampling2(vec2 uv)
{
    const float T_RES = 32.0;
    return vec3(gnoise(uv*T_RES)); //讀取亂數函式
}

float triplanarSampling(vec3 p, vec3 n)
{
    float fTotal = abs(n.x)+abs(n.y)+abs(n.z);
    return  (abs(n.x)*smoothSampling2(p.yz).x
            +abs(n.y)*smoothSampling2(p.xz).x
            +abs(n.z)*smoothSampling2(p.xy).x)/fTotal;
}

const mat2 m2 = mat2(0.90,0.44,-0.44,0.90);
float triplanarNoise(vec3 p, vec3 n)
{
    const float BUMP_MAP_UV_SCALE = 0.2;
    float fTotal = abs(n.x)+abs(n.y)+abs(n.z);
    float f1 = triplanarSampling(p*BUMP_MAP_UV_SCALE,n);
    p.xy = m2*p.xy;
    p.xz = m2*p.xz;
    p *= 2.1;
    float f2 = triplanarSampling(p*BUMP_MAP_UV_SCALE,n);
    p.yx = m2*p.yx;
    p.yz = m2*p.yz;
    p *= 2.3;
    float f3 = triplanarSampling(p*BUMP_MAP_UV_SCALE,n);
    return f1+0.5*f2+0.25*f3;
}

vec3 normalMap(vec3 p, vec3 n)
{
    float d = 0.005;
    float po = triplanarNoise(p,n);
    float px = triplanarNoise(p+vec3(d,0,0),n);
    float py = triplanarNoise(p+vec3(0,d,0),n);
    float pz = triplanarNoise(p+vec3(0,0,d),n);
    return normalize(vec3((px-po)/d,
                          (py-po)/d,
                          (pz-po)/d));
}
/* === end === */
