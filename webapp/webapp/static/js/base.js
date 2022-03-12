// menu button to toggle sidebar - collapse or expand
let menu_btn = document.querySelector(".sidebar .top .bx");
let grid_container = document.querySelector("body .grid-container");
menu_btn.onclick = function() {
    grid_container.classList.toggle("sidebar-expanded");
}

let current_theme = "dark";
let theme_btn = document.querySelector(".sidebar #light-dark-theme-icon");
theme_btn.onclick = function() {
    let css_theme = document.createElement("link");
    current_theme = (current_theme == "light") ? "dark" : "light";
    css_theme.href = '/static/css/theme_' + current_theme + '.css';
    css_theme.rel = "stylesheet";
    document.getElementsByTagName("head")[0].appendChild(css_theme);
}