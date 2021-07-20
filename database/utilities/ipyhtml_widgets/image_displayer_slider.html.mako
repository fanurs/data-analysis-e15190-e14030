<!-- ========== ${uniqtag} START ========== -->
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script type="text/javascript" id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>

<!-- HTML for ${uniqtag} -->
<div id="${uniqtag}-div">
  <div id="${uniqtag}-controls-div">
    <div id="${uniqtag}-slider-label-div">
      % for ii, label in enumerate(figures.keys()):
      <button id="${uniqtag}-slider-label${ii}" class="${uniqtag}-slider-label" style="display: none;">${label}</button>
      % endfor
    </div>
    <div id="${uniqtag}-button-slider-div">
      <button id="${uniqtag}-button-left" class="${uniqtag}-button-unclicked" onclick="${uniqtag}_button_click('left')">
        <i class="${uniqtag}-arrow ${uniqtag}-arrow-left"></i>
      </button>
      <input type="range" id="${uniqtag}-slider" min=0 max=${len(figures) - 1} value=${first} oninput="${uniqtag}_slider_action()" style="${slider_style}">
      <button id="${uniqtag}-button-right" class="${uniqtag}-button-unclicked" onclick="${uniqtag}_button_click('right')">
        <i class="${uniqtag}-arrow ${uniqtag}-arrow-right"></i>
      </button>
    </div>
    <div id="${uniqtag}-animation-div">
      <button id="${uniqtag}-button-play-pause" onclick="${uniqtag}_play_pause_click()">?</button>
      <select id="${uniqtag}-animation-speed" onclick="${uniqtag}_update_speed(this.value)">
        <option value=10.00>10x</option>
        <option value=5.00>5x</option>
        <option value=2.00>2x</option>
        <option selected value=1.00>1x</option>
        <option value=0.50>0.5x</option>
        <option value=0.2>0.2x</option>
        <option value=0.1>0.1x</option>
      </select>
    </div>
  </div>
  <div id="${uniqtag}-div-images">
    % for ii, image in enumerate(figures.values()):
    <img id="${uniqtag}-img${ii}" class="${uniqtag}-img" src="${image}" style="${image_style} display: none;">
    % endfor
  </div>
</div>

<!-- CSS for ${uniqtag} -->
<style>
  #${uniqtag}-controls-div {
    display: table;
    justify-content: center;
    vertical-align: middle;
  }
  #${uniqtag}-slider-label-div {
    display: flex;
    justify-content: left;
    vertical-align: middle;
  }
  #${uniqtag}-button-slider-div {
    display: flex;
    justify-content: left;
    vertical-align: middle;
  }
  #${uniqtag}-animation-div {
    display: table-cell;
    padding-left: 30px;
    vertical-align: middle;
    height: 30px;
  }
  #${uniqtag}-button-play-pause {
    border-radius: 10px;
    width: 40px;
    line-height: 20px;
    padding: 8px 0px;
    color: black;
    background-color: white;
    border-color: black;
    border-width: 3px;
  }
  #${uniqtag}-button-play-pause:hover {
    background-color: gainsboro;
    box-shadow: 1px 1px 3px black;
    transition: all 0.2s ease-in-out;
  }
  .${uniqtag}-slider-label {
    display: inline-block;
    color: black;
    background: palegreen;
    justify-content: center;
    padding: 4px 15px;
    border-radius: 20px;
    border-width: 3px;
    height: 40px;
    font-size: 20px;
    font-weight: bold;
    transition: all 0.2s ease-in-out;
  }
  .${uniqtag}-slider-label:active {
    border-style: outset;
  }
  #${uniqtag}-slider, .${uniqtag}-button-unclicked {
    margin-top: 5px;
    height: 30px;
  }
  .${uniqtag}-button-unclicked {
    vertical-align: middle;
    display: inline-flex;
    align-items: center;
    background: wheat;
    padding: 10px 15px;
    border-radius: 10px;
    transition: all 0.2s ease-in-out;
  }
  .${uniqtag}-button-unclicked:hover {
    background: gold;
    box-shadow: 1px 1px 2px black;
  }
  .${uniqtag}-button-unclicked:active {
    box-shadow: 0 10px 50px gold;
  }
  .${uniqtag}-arrow {
    border: solid black;
    border-width: 0 3px 3px 0;
    display: inline-block;
    padding: 3px;
  }
  .${uniqtag}-arrow-left {
    transform: rotate(135deg);
    -webkit-transform: rotate(135deg);
    margin-left: 4px;
  }
  .${uniqtag}-arrow-right {
    transform: rotate(-45deg);
    -webkit-transform: rotate(-45deg);
    margin-right: 4px;
  }
  #${uniqtag}-slider {
    -webkit-appearance: none; /* to enable custom slider design */
    background-color: transparent;
    display: inline-flex;
    vertical-align: middle;
    width: 300px;
    margin-left: 3px;
    margin-right: 3px;
  }
  #${uniqtag}-slider::-webkit-slider-thumb {
    -webkit-appearance: none;
    border: 5px;
    border-color: black;
    border-radius: 10px;
    height: 30px;
    width: 50px;
    margin-top: -8px;
    margin-bottom: -8px;
    background: navajowhite;
    cursor: pointer;
  }
  #${uniqtag}-slider::-webkit-slider-thumb:hover {
    -webkit-appearance: none;
    background: gold;
    box-shadow: 0px 10px 50px gold, 1px 1px 2px black;
  }
  #${uniqtag}-slider::-webkit-slider-runnable-track {
    background: cornsilk;
    border-radius: 2px;
    border-color: black;
  }
  .${uniqtag}-img {
    height: auto;
    width: auto;
    margin-top: 2px !important;
  }
</style>

<!-- JavaScript for ${uniqtag} -->
<script>
  function ${uniqtag}_slider_action() {
    var slider = document.getElementById("${uniqtag}-slider");
    for (var k = 0; k < ${len(figures)}; ++k) {
      document.getElementById("${uniqtag}-img" + k).style.display = "none";
    document.getElementById("${uniqtag}-slider-label" + k).style.display = "none";
    }
    var i = parseInt(slider.value, 10);
    document.getElementById("${uniqtag}-img" + i).style.display = "block";
    document.getElementById("${uniqtag}-slider-label" + i).style.display = "block";
  }

  function ${uniqtag}_button_click(direction) {
    var slider = document.getElementById("${uniqtag}-slider");
    if (direction == "left") {
      var new_value = ((parseInt(slider.value, 10) - 1) + ${len(figures)}) % ${len(figures)};
    }
    else if (direction == "right") {
      var new_value = ((parseInt(slider.value, 10) + 1) + ${len(figures)}) % ${len(figures)};
    }
    document.getElementById(slider.id).value = new_value;
    ${uniqtag}_slider_action();
  }

  var unicode_play = "&#x25B6";
  var unicode_pause = "&#x23F8";
  var ${uniqtag}_animation_innerHTML = null;
  var ${uniqtag}_animation_action = null;
  const ${uniqtag}_animation_interval_base = 1000; // millisecond
  var ${uniqtag}_animation_interval = 1000; // millisecond
  function ${uniqtag}_play_pause_click() {
    var id = "${uniqtag}-button-play-pause";
    var button = document.getElementById(id);
    if (${uniqtag}_animation_innerHTML == "play") {
      ${uniqtag}_animation_action = setInterval(function() {${uniqtag}_button_click("right");}, ${uniqtag}_animation_interval);
      document.getElementById(id).style.backgroundColor = "gainsboro";
      document.getElementById(id).innerHTML = unicode_pause;
      ${uniqtag}_animation_innerHTML = "pause";
    }
    else if (${uniqtag}_animation_innerHTML == "pause") {
      clearInterval(${uniqtag}_animation_action);
      document.getElementById(id).innerHTML = unicode_play;
      document.getElementById(id).style.backgroundColor = "white";
      ${uniqtag}_animation_innerHTML = "play";
    }
  }

    function ${uniqtag}_update_speed(multiplier) {
      ${uniqtag}_animation_interval = parseInt(${uniqtag}_animation_interval_base / multiplier, 10);
      ${uniqtag}_play_pause_click();
      ${uniqtag}_play_pause_click();
    }

  // initialization
  document.getElementById("${uniqtag}-button-play-pause").innerHTML = unicode_play;
  ${uniqtag}_animation_innerHTML = "play";
  ${uniqtag}_slider_action();
</script>
<!-- ========== ${uniqtag} END ========== -->