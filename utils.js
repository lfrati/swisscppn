// TODO: remove hardcoded tex/row names?
function make_init(params) {
  let body = [];
  let cnt = 0;
  for (let i in params) {
    p = params[i];
    switch (p[0]) {
      case "w":
        body.push(`get_mat4(row, ${cnt})`);
        cnt += 4;
        break;
      case "b":
        body.push(`get_vec4(row, ${cnt})`);
        cnt += 1;
        break;
      default:
        console.error("UNKNOWN value:", p);
    }
  }
  let init = `Net(${body.join(",\n    ")}\n)`;
  return [init, cnt + 1];
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

/**
 *
 * struct Net{
 *   mat4 w1_0;
 *   vec4 b1_0;
 *   mat4 w1_1;
 *   vec4 b1_1;
 *   mat4 w3_0;
 *   vec4 b3_0;
 *   mat4 w3_1;
 *   vec4 b3_1;
 *   mat4 w4_0;
 *   vec4 b4_0;
 * };
 *
 * Net(get_mat4(tex, row, 0),
 *   get_vec4(tex, row, 4),
 *   get_mat4(tex, row, 5),
 *   get_vec4(tex, row, 9),
 *   get_mat4(tex, row, 10),
 *   get_vec4(tex, row, 14),
 *   get_mat4(tex, row, 15),
 *   get_vec4(tex, row, 19),
 *   get_mat4(tex, row, 20),
 *   get_vec4(tex, row, 24)
 * );
 *
 * vec4 f(vec4 inputs, Net n){
 *   vec4 h1_1 = relu_norm(inputs * n.w1_1 + n.b1_1);
 *   vec4 h1_2 = relu_norm(inputs * n.w1_2 + n.b1_2);
 *   vec4 h1 = h1_1 + h1_2 ;
 *   vec4 h2_1 = cos((h1 * n.w2_2)*0.5 + n.b2_2);
 *   vec4 h2_2 = cos((h1 * n.w2_1)*0.5 + n.b2_1);
 *   vec4 h2 = h2_1 + h2_2 ;
 *   return fsig(h2 * n.w3);
 * }
 *
 *
 */

function parse(conf) {
  /*
   * All weights are 4x4, all biases are 4x1
   */
  let forward = ["vec4 forward(in vec4 x0, in Net net){"];
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
          if (affine) {
            let bias = `b${depth}_${branch}`;
            params.push(bias);
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
        if (affine) {
          let bias = `b${depth}_0`;
          params.push(bias);
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

  let decl = [];
  for (let val of params) {
    decl.push(val[0] === "w" ? `  mat4 ${val};` : `  vec4 ${val};`);
  }
  decl = ["struct Net{\n" + decl.join("\n") + "\n};\n"];

  let code = decl.concat(forward);
  return [code.join("\n"), params];
}
