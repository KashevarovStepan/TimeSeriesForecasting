import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.DateFormat;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Logger;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

public class TimeSeriesForecasting {

  private static final Logger log = Logger.getLogger(TimeSeriesForecasting.class.getName());
  private static final String path = "./resources/datasetExtended.csv";//1-1258    0-1257
  private static final String fpath = "./resources/forecast.csv";
  private static final Integer t = 999;
  private static final Integer P = 1257 - t;
  private static final Integer offset = 10;
  private static final Integer samples = 100;
  private static final Integer degree = 3;
  private static final Double lucky = 1e4;
  private static final Double eps = 1e5;
  private static final Double mutationDepth = 0.5;

  private static class pair {

    LocalDate date;
    Double price;

    public pair(LocalDate date, Double price) {
      this.date = date;
      this.price = price;
    }
  }

  private static List<pair> makeDataSet() {
    List<pair> data = new ArrayList<>();
    try (BufferedReader reader = new BufferedReader(new FileReader(path))) {
      String line = reader.readLine();
      while ((line = reader.readLine()) != null) {
        String[] values = line.split(",");
        data.add(new pair(LocalDate.parse(values[0]), Double.parseDouble(values[1])));
      }
    } catch (IOException exception) {
      log.info(exception.getMessage());
    }
    return data;
  }

  public static String escapeSpecialCharacters(String data) {
    String escapedData = data.replaceAll("\\R", " ");
    if (data.contains(",") || data.contains("\"") || data.contains("'")) {
      data = data.replace("\"", "\"\"");
      escapedData = "\"" + data + "\"";
    }
    return escapedData;
  }

  private static String convertToCSV(String[] data) {
    return Stream.of(data)
        .map(data1 -> escapeSpecialCharacters(data1))
        .collect(Collectors.joining(","));
  }

  public static void givenDataArray_whenConvertToCSV_thenOutputCreated(List<String[]> dataLines)
      throws IOException {
    File csvOutputFile = new File(fpath);
    try (PrintWriter pw = new PrintWriter(csvOutputFile)) {
      dataLines.stream()
          .map(data -> convertToCSV(data))
          .forEach(pw::println);
    }
  }

  private static Double poly(List<Double> c, Double x) {
    Double sum = 0d;
    for (int i = 0; i < c.size(); i++) {
      sum += c.get(i) * Math.pow(x, i);
    }
    return sum;
  }

  private static Double bestFitnessOfSorted(List<List<Double>> coef, List<pair> data,
      List<pair> present, Integer startIndex) {
    Double fitness = 0d;
    for (int j = 0; j < offset; j++) {
      fitness += Math.pow(poly(coef.get(0), data.get(startIndex + j).price)
          - present.get(j).price, 2);
    }
    return fitness;
  }

  private static List<Double> fitnessOf(List<List<Double>> coef, List<pair> data,
      List<pair> present, Integer startIndex) {
    List<Double> fit = new ArrayList<>();
    Double fitness;
    for (int i = 0; i < coef.size(); i++) {
      fitness = 0d;
      for (int j = 0; j < offset; j++) {
        fitness += Math.pow(poly(coef.get(i), data.get(startIndex + j).price)
            - present.get(j).price, 2);
      }
      fit.add(fitness);
    }
    return fit;
  }

  private static List<List<Double>> sortPoly(List<List<Double>> coef, List<pair> data,
      List<pair> present, Integer startIndex) {
    Double tmp;
    List<Double> temp;
    List<Double> fitness = fitnessOf(coef, data, present, startIndex);
    for (int i = 0; i < coef.size(); i++) {
      for (int j = i + 1; j < coef.size(); j++) {
        if (fitness.get(i) > fitness.get(j)) {
          tmp = fitness.get(i);
          fitness.set(i, fitness.get(j));
          fitness.set(j, tmp);
          temp = coef.get(i);
          coef.set(i, coef.get(j));
          coef.set(j, temp);
        }
      }
    }
    return coef;
  }

  private static List<List<Double>> newGeneration(List<List<Double>> coef, List<pair> data,
      List<pair> present, Integer startIndex,
      Double gmd) {
    List<List<Double>> newC = new ArrayList<>();
    //Breeding
    for (int i = coef.size() / 2; i < coef.size() * 3 / 4; i++) {
      Integer crossingPoint = ThreadLocalRandom.current().nextInt(0, degree);
      List<Double> parentNumberOne = coef.get(2 * (i - coef.size() / 2));
      List<Double> parentNumberTwo = coef.get(2 * (i - coef.size() / 2) + 1);
      List<Double> c = new ArrayList<>();
      for (int j = 0; j <= degree; j++) {
        /*if (j <= crossingPoint) {
          c.add(parentNumberOne.get(j));
        } else {
          c.add(parentNumberTwo.get(j));
        }*/
        double tmp = parentNumberTwo.get(j) + Math.random()
            * (parentNumberOne.get(j) - parentNumberTwo.get(j));// t*x + (1-t)*y = y + t*(x-y)
        c.add(tmp);
      }
      coef.set(i, c);
    }
    /*for (int i = 0; i < coef.size() / 4; i++) {
      Integer crossingPoint = ThreadLocalRandom.current().nextInt(0, degree);
      List<Double> parentNumberOne = coef.get(2 * i);
      List<Double> parentNumberTwo = coef.get(2 * i + 1);
      List<Double> c = new ArrayList<>();
      for (int j = 0; j <= degree; j++) {
        if (j <= crossingPoint) {
          c.add(parentNumberOne.get(j));
        } else {
          c.add(parentNumberTwo.get(j));
        }
      }
      newC.add(c);
    }*/
    //Mutating
    for (int i = coef.size() * 3 / 4; i < coef.size() * 7 / 8; i++) {
      List<Double> c = new ArrayList<>();
      c.addAll(coef.get(i - coef.size() / 4));
      for (int j = 0; j < coef.get(0).size(); j++) {
        if (mutationDepth > Math.random()) {
          c.set(j, c.get(j) + (Math.random() - 0.5) * c.get(j) / 10.0);
        }
      }
      /*Integer pointMut = ThreadLocalRandom.current().nextInt(0, degree + 1);
      c.set(pointMut, -c.get(pointMut));*/
      coef.set(i, c);
    }
    /*for (int i = coef.size() / 4; i < coef.size() * 3 / 4; i++) {
      List<Double> c = new ArrayList<>();
      if (i < coef.size() / 2) {
        c.addAll(newC.get(i - coef.size() / 4));
      } else {
        c.addAll(coef.get(i - coef.size() / 2));
      }
      Integer pointMut = ThreadLocalRandom.current().nextInt(0, degree + 1);
      c.set(pointMut, c.get(pointMut) + (Math.random() - 0.5) * gmd /2);
      newC.add(c);
    }*/
    //New random
    for (int i = coef.size() * 7 / 8; i < coef.size(); i++) {
      List<Double> c = new ArrayList<>();
      for (int j = 0; j <= degree; j++) {
        c.add((Math.random() - 0.5) * lucky);//TYT
      }
      coef.set(i, c);
    }
    /*for (int i = coef.size() * 3 / 4; i < coef.size() * 7 / 8; i++) {
      List<Double> c = new ArrayList<>();
      for (int j = 0; j <= degree; j++) {
        c.add((Math.random()-0.5)*gmd/2);//TYT
      }
      newC.add(c);
    }
    for (int i = coef.size() * 7 / 8; i < coef.size(); i++) {
      newC.add(coef.get(i - coef.size() * 7 / 8));
    }*/
    coef = sortPoly(coef, data, present, startIndex);
    return coef;
  }

  private static List<pair> GA(List<pair> data, List<pair> present, Integer startIndex,
      Double gmd) {
    Double fitness;
    Integer counter = 0;
    List<pair> res = new ArrayList<>();
    List<List<Double>> coef = new ArrayList<>();
    for (int j = 0; j < samples * 10; j++) {
      List<Double> c = new ArrayList<>();
      for (int i = 0; i <= degree; i++) {
        c.add((Math.random() - 0.5) * lucky);//TYT
      }
      coef.add(c);
    }
    log.info("Polynomials generated");
    Double dif = 0d;
    coef = sortPoly(coef, data, present, startIndex);
    do {
      coef = newGeneration(coef, data, present, startIndex, gmd);
      log.info("New generation");
      fitness = bestFitnessOfSorted(coef, data, present, startIndex);
      log.info("Iteration = " + (counter++) + "       Fitness = " + fitness.toString()
          + "    dif = " + Math.abs(fitness - dif));
      dif = fitness;
    } while (counter < 10000);//(fitness>eps)!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    List<Double> g = coef.get(0);
    for (int i = 0; i < offset; i++) {
      LocalDate cur = present.get(offset - 1).date;
      for (int j = 0; j <= i; j++) {
        cur = cur.plusWeeks(1);
      }
      res.add(new pair(cur, poly(g, data.get(startIndex + offset + i).price)));
    }
    return res;
  }

  private static List<Integer> patternOf(List<pair> data, Integer start) {
    List<Integer> patter = new ArrayList<>();
    for (int i = start; i < start + offset; i++) {
      if (start == 0) {
        patter.add(1);
      } else {
        if (data.get(i - 1).price > data.get(i).price) {
          patter.add(0);
        } else {
          patter.add(1);
        }
      }
    }
    return patter;
  }

  private static Integer compPatterns(List<Integer> pattern1, List<Integer> pattern2) {
    Integer counter = 0;
    for (int i = 0; i < pattern1.size(); i++) {
      if (pattern1.get(i) == pattern2.get(i)) {
        counter++;
      }
    }
    return counter;
  }

  private static List<pair> forecasting(List<pair> data) {
    log.info("Started");
    Integer bestPatternIndex = 0;
    Double _gmd = (double) offset * (offset - 1);
    Double _tau = (double) offset * (offset - 1);
    Double _rho = (double) offset * (offset * offset - 1);
    List<List<Double>> stat = new ArrayList<>();//tau,gmd,rho;
    List<pair> res;
    List<pair> present = data.subList(data.size() - offset, data.size());
    List<Integer> presentPattern = new ArrayList<>();
    List<Integer> bestPattern = patternOf(data, 0);
    List<Integer> pattern = new ArrayList<>();
    List<Integer> indexes = new ArrayList<>();
    if (data.get(data.size() - offset - 1).price > present.get(0).price) {
      presentPattern.add(0);
    } else {
      presentPattern.add(1);
    }
    for (int i = 1; i < present.size(); i++) {
      if (present.get(i - 1).price > present.get(i).price) {
        presentPattern.add(0);
      } else {
        presentPattern.add(1);
      }
    }
    List<Double> val = new ArrayList<>();
    for (pair cur : data) {
      val.add(cur.price);
    }
    for (int i = 0; i < t - 2 * offset; i++) {
      pattern.clear();
      Double tau = 0d, gmd = 0d, rho = 0d;
      for (int j = 0; j < offset; j++) {
        rho -= Math.pow(present.get(j).price - val.get(i + j), 2);
        for (int k = 0; k < offset; k++) {
          if (k != j) {
            gmd += Math.abs(val.get(i + j) - val.get(i + k));
          }
        }
        for (int k = j + 1; k < offset; k++) {
          if ((present.get(j).price - present.get(k).price)
              * (val.get(i + j) - val.get(i + k)) > 0) {
            tau += 1;
          } else {
            tau -= 1;
          }
        }
      }
      tau = tau * 2 / _tau;
      if (tau > 1 || tau < -1) {
        log.info("tau[" + i + "] = " + tau.toString());
      }
      gmd /= _gmd;
      if (gmd < 0) {
        log.info("gmd[" + i + "] = " + gmd.toString());
      }
      rho = rho * 6 / _rho + 1;
      if (rho > 1) {
        log.info("rho[" + i + "] = " + rho.toString());
      }
      stat.add(List.of(new Double[]{tau, gmd, rho}));
      pattern = patternOf(data, i);
      if (compPatterns(pattern, presentPattern) > compPatterns(bestPattern, presentPattern)) {
        bestPattern = pattern;
        bestPatternIndex = i;
      }
    }
    log.info("Calculated tau,gmd,rho");
    Double worst;
    Integer worstIndex;
    for (int i = 0; i < stat.size(); i++) {
      if (indexes.size() < samples) {
        indexes.add(i);
      } else {
        worst = 1d;
        worstIndex = -1;
        for (int j = 0; j < indexes.size(); j++) {
          if (stat.get(indexes.get(j)).get(0) < worst) {
            worst = stat.get(indexes.get(j)).get(0);
            worstIndex = j;
          }
        }
        if (worst < stat.get(i).get(0)) {
          indexes.set(worstIndex, i);
        }
      }
    }
    log.info("Found matches");
    Integer bestIndex = -1;
    Double best = -1d;
    for (Integer cur : indexes) {
      if (stat.get(cur).get(0) > best) {
        best = stat.get(cur).get(0);
        bestIndex = cur;
      }
    }
    log.info("Found best by tau");
    //bestIndex=bestPatternIndex;//!!!!!!!!!!!!!!!!!!
    res = GA(data, present, bestIndex, stat.get(bestIndex).get(1));
    return res;
  }

  private static List<pair> makeForecast(List<pair> data) throws IOException {
    List<pair> res = new ArrayList<>();
    //res=data.subList(t+1,t+P+1);
    res = forecasting((data.subList(0, t + 1)));

    List<String[]> dataLines = new ArrayList<>();
    for (pair cur : res) {
      String[] tmp = new String[]{cur.date.toString(), cur.price.toString()};
      dataLines.add(tmp);
    }
    givenDataArray_whenConvertToCSV_thenOutputCreated(dataLines);
    return res;
  }

  public static void main(String[] args) throws IOException {
    List<pair> data = makeDataSet();
    List<pair> forecast = makeForecast(data);
    log.info("AAAAA");
  }
}
