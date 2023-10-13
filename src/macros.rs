#[macro_export]
macro_rules! mat {
  ( $( $( $x:expr),+ );+ ) => {
      {
          let mut data = Vec::new();
          let mut rows = 0;
          let mut cols = 0;

          $(
              rows += 1;
              cols = 0;
              $(
                  data.push($x);
                  cols += 1;
              )+
          )+

        Matrix { rows, cols, data }
      }
  };
}
