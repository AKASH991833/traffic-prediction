module.exports.run = async (page, base, shot) => {
  const out = [];
  await page.waitForFunction(() => document.querySelector('#best').textContent.includes('M'), {timeout: 8000});
  const best = await page.$eval('#best', e => e.textContent);
  out.push(['best window: ' + best, /\d{1,2}:\d{2} [AP]M - \d{1,2}:\d{2} [AP]M/.test(best)]);
  const mape = await page.$eval('#m-mape', e => e.textContent);
  const mv = parseFloat(mape);
  out.push(['MAPE sane: ' + mape, mv > 1 && mv < 45]);
  const cells = await page.$$eval('.hcell', r => r.length);
  out.push(['heatmap 168 cells', cells === 168]);
  const save = await page.$eval('#save', e => e.textContent);
  out.push(['savings line', /less congestion/.test(save)]);
  await page.select('#day', '5'); // Saturday
  await new Promise(r=>setTimeout(r,400));
  const best2 = await page.$eval('#best', e => e.textContent);
  out.push(['re-render on day change', /\d{1,2}:\d{2}/.test(best2)]);
  if (shot) await page.screenshot({path: shot});
  return out;
};
