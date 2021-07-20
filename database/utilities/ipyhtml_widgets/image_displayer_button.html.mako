<!-- ========== ${uniqtag} START ========== -->
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script type="text/javascript" id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>

<!-- HTML for ${uniqtag} -->
<div id="${uniqtag}-div">
  <div id="${uniqtag}-div-controls">
    % for ii, label in enumerate(figures.keys()):
    <button id="${uniqtag}-button${ii}" class="${uniqtag}-button ${uniqtag}-button-unclicked" onclick="${uniqtag}_button_click(${ii})">${label}</button>
    % endfor
  </div>
  <div id="${uniqtag}-div-images">
    % for ii, image in enumerate(figures.values()):
    <img id="${uniqtag}-img${ii}" class="${uniqtag}-img" src="${image}" style="${image_style} display: none;">
    % endfor
  </div>
</div>

<!-- CSS for ${uniqtag} -->
<style>
  #${uniqtag}-div-controls {
    display: flex;
    align-items: center;
  }
  #${uniqtag}-div-images {
    display: flex;
    align-items: center;
  }
  .${uniqtag}-img {
    height: auto;
    width: auto;
    margin-top: 2px !important;
  }
  .${uniqtag}-button {
    color: black;
    font-size: 20px;
    height: 40px;
    margin: 0px 3px;
    transition: all 0.2s ease-in-out;
  }
  .${uniqtag}-button-unclicked {
    background: wheat;
    padding: 5px 15px;
    border-radius: 10px;
    font-weight: normal;
  }
  .${uniqtag}-button-clicked {
    background: palegreen;
    padding: 4px 15px;
    border-radius: 20px;
    border-width: 3px;
    font-weight: bold;
  }
  .${uniqtag}-button-unclicked:hover:not(.active) {
    background: gold;
    box-shadow: 0 10px 50px gold, 1px 1px 2px black;
  }
  .${uniqtag}-button-unclicked:hover:active {
    background: yellowgreen;
    box-shadow: 0 10px 50px green, 1px 1px 2px black;
  }
  .${uniqtag}-button-clicked:active {
    border-style: outset;
  }
  .${uniqtag}-button-clicked:hover {
    background: palegreen;
    box-shadow: 0 10px 25px darkgreen;
  }
</style>

<!-- JavaScript for ${uniqtag} -->
<script>
  function ${uniqtag}_button_click(i) {
    for (var k = 0; k < ${len(figures)}; ++k) {
      document.getElementById("${uniqtag}-img" + k).style.display = "none";
      document.getElementById("${uniqtag}-button" + k).classList.remove("${uniqtag}-button-clicked");
      document.getElementById("${uniqtag}-button" + k).classList.add("${uniqtag}-button-unclicked");
    }
    document.getElementById("${uniqtag}-img" + i).style.display = "block";
    document.getElementById("${uniqtag}-button" + i).classList.remove("${uniqtag}-button-unclicked");
    document.getElementById("${uniqtag}-button" + i).classList.add("${uniqtag}-button-clicked");
  }

  // initialization
  ${uniqtag}_button_click(${first});
</script>
<!-- ========== ${uniqtag} END ========== -->