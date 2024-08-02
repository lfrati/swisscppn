function minit() {
  return (Math.random() * 2 - 1) * 1.5;
}

function binit() {
  return Math.random();
}

function mat4init() {
  return `mat4(${minit()},${minit()},${minit()},${minit()},${minit()},${minit()},${minit()},${minit()},${minit()},${minit()},${minit()},${minit()},${minit()},${minit()},${minit()},${minit()})`;
}

function vec4init() {
  return `vec4(${binit()},${binit()},${binit()},${binit()})`;
}

function make_init(params) {
  let init = ["Net("];
  for (let i in params) {
    p = params[i];
    if (i > 0) {
      init.push(", ");
    }
    if (p[0] == "w") init.push(`${mat4init()}`);
    else init.push(`${vec4init()}`);
  }
  init.push(")");
  return init.join("");
}

function make_net(params) {
  net = "struct Net{\n";
  for (let p of params) {
    if (p[0] == "w") net += `  mat4 ${p};\n`;
    else net += `  vec4 ${p};\n`;
  }
  net += "};\n";
  return net;
}

function make_lerp(params) {
  lerp_fun = "Net lerp(in Net n1, in Net n2, in float t){\n  Net n3;\n";
  for (let p of params) {
    lerp_fun += `  n3.${p} = lerp(n1.${p}, n2.${p}, t);\n`;
  }
  lerp_fun += "  return n3;\n}\n";
  return lerp_fun;
}

/**
vec4 f(vec4 inputs, Net n){
  vec4 h1_1 = relu_norm(inputs * n.w1_1 + n.b1_1);
  vec4 h1_2 = relu_norm(inputs * n.w1_2 + n.b1_2);
  vec4 h1 = h1_1 + h1_2 ;
  vec4 h2_1 = cos((h1 * n.w2_2)*0.5 + n.b2_2);
  vec4 h2_2 = cos((h1 * n.w2_1)*0.5 + n.b2_1);
  vec4 h2 = h2_1 + h2_2 ;
  return fsig(h2 * n.w3);
}

mat4 lerp(mat4 w1, mat4 w2, float t){return w1 + (w2 - w1)*t;}
vec4 lerp(vec4 w1, vec4 w2, float t){return w1 + (w2 - w1)*t;}
void lerp(inout Net n1, in Net n2, const float t){
          n1.w1_1 = lerp(n1.w1_1, n2.w1_1, t);
          n1.w1_2 = lerp(n1.w1_2, n2.w1_2, t);
          n1.w2_1 = lerp(n1.w2_1, n2.w2_1, t);
          n1.w2_2 = lerp(n1.w2_2, n2.w2_2, t);
          n1.w3 =  lerp(n1.w3, n2.w3, t);
          n1.b1_1 = lerp(n1.b1_1, n2.b1_1, t);
          n1.b1_2 = lerp(n1.b1_2, n2.b1_2, t);
          n1.b2_1 = lerp(n1.b2_1, n2.b2_1, t);
          n1.b2_2 = lerp(n1.b2_2, n2.b2_2, t);
}
*/

function gen_lerp(val) {
  return `  n1.${val} = lerp(n1.${val}, n2.${val}, t);`;
}

function parse(conf) {
  /*
   * All weights are 4x4 all biases are 4x1
   */
  let forward = ["vec4 forward(in vec4 x0, in Net net){"];
  let lerp = [
    `mat4 lerp(mat4 w1, mat4 w2, float t){return w1 + (w2 - w1)*t;}
vec4 lerp(vec4 w1, vec4 w2, float t){return w1 + (w2 - w1)*t;}
void lerp(inout Net n1, in Net n2, const float t){`,
  ];
  let params = [];
  let depth = 0;
  let inp, out;
  for (let layer of conf) {
    let width = layer.width || 0;
    let affine = layer.affine || true;
    console.log(layer, width);
    depth += 1;
    inp = depth - 1;
    out = depth;

    switch (layer.type) {
      case "par":
        let partials = [];
        // STEP 1: parallel branches
        for (let branch = 0; branch < layer.width; branch++) {
          let partial = `x${depth}_${branch}`;
          partials.push(partial);
          let weight = `w${depth}_${branch}`;
          params.push(weight);
          lerp.push(gen_lerp(weight));
          lerp.push();
          if (affine) {
            let bias = `b${depth}_${branch}`;
            params.push(bias);
            lerp.push(gen_lerp(bias));
            forward.push(
              `  vec4 ${partial} = ${layer.f}(x${inp} * net.${weight} + net.${bias});`
            );
          } else {
            forward.push(
              `  vec4 ${partial} = ${layer.f}(x${inp} * net.${weight});`
            );
          }
        }
        // STEP 2: summation
        forward.push(`  vec4 x${out} = ${partials.join(" + ")};`);
        break;
      case "lin":
        // Same as par with width=1 but we implement it separately
        // to avoid creating intermediate values
        let weight = `w${depth}_0`;
        params.push(weight);
        lerp.push(gen_lerp(weight));
        if (affine) {
          let bias = `b${depth}_0`;
          params.push(bias);
          lerp.push(gen_lerp(bias));
          forward.push(
            `  vec4 x${out} = ${layer.f}(x${inp} * net.${weight} + net.${bias});`
          );
        } else {
          forward.push(`  vec4 x${out} = ${layer.f}(x${inp} * net.${weight});`);
        }
        break;
      case "f":
        forward.push(`  vec4 x${out} = ${layer.f}(x${inp});`);
        break;
      default:
        console.error("Unknown layer:", layer.type);
    }
  }
  forward.push(`  return x${out};\n}\n`);
  lerp.push("}");

  let decl = [];
  for (let val of params) {
    decl.push(val[0] === "w" ? `  mat4 ${val};` : `  vec4 ${val};`);
  }
  decl = ["struct Net{\n" + decl.join("\n") + "\n};\n"];

  let code = decl.concat(forward).concat(lerp);
  return [code.join("\n"), params];
}
