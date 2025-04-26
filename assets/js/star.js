// 增强版流星效果
class StarParticle {
  constructor(x, y) {
    this.element = document.createElement('div');
    this.element.className = 'star-particle';
    this.x = x;
    this.y = y;
    this.angle = Math.random() * Math.PI * 2;
    this.radius = 20 + Math.random() * 30;
    this.speed = 0.05 + Math.random() * 0.05;
    this.life = 1;
    
    document.body.appendChild(this.element);
  }

  update() {
    this.angle += this.speed;
    const dx = Math.cos(this.angle) * this.radius;
    const dy = Math.sin(this.angle) * this.radius;
    
    this.element.style.transform = `
      translate(${this.x + dx}px, ${this.y + dy}px)
      scale(${this.life})
    `;
    this.life -= 0.01;
    
    if (this.life <= 0) {
      this.element.remove();
      return false;
    }
    return true;
  }
}

// 主逻辑
document.addEventListener('DOMContentLoaded', () => {
  const particles = [];
  
  // 创建初始粒子
  for (let i = 0; i < 8; i++) {
    particles.push(new StarParticle(0, 0));
  }

  // 动画循环
  function animate() {
    particles.forEach((p, index) => {
      if (!p.update()) {
        particles.splice(index, 1);
      }
    });
    requestAnimationFrame(animate);
  }
  animate();

  // 鼠标移动更新位置
  document.addEventListener('mousemove', (e) => {
    particles.forEach(p => {
      p.x = e.clientX;
      p.y = e.clientY;
    });

    // 保留原有流星轨迹逻辑
    const trail = document.createElement('div');
    trail.className = 'trail';
    trail.style.left = e.pageX - 3 + 'px';
    trail.style.top = e.pageY - 3 + 'px';
    trail.style.background = `hsla(${Math.random()*360}, 70%, 50%, 0.8)`;
    
    document.body.appendChild(trail);
    setTimeout(() => trail.remove(), 1000);
  });
});